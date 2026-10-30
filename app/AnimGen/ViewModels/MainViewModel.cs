using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using AnimGen.Services;
using Microsoft.Win32;

namespace AnimGen.ViewModels;

/// <summary>Kinds of status line. Each one has its own icon and wording, not only its own colour (§11.1).</summary>
public enum StatusKind { Info, Busy, Ok, Warning, Error }

/// <summary>A hanging part the owner can stop from swinging (automatic detection, corrected by the owner).</summary>
public sealed class HangingItem : ObservableObject
{
    private readonly Func<HangingItem, Task> _changed;
    private bool _on;

    public HangingItem(string name, bool on, Func<HangingItem, Task> changed)
    {
        Name = name;
        _on = on;
        _changed = changed;
    }

    public string Name { get; }

    public string Label => Name switch
    {
        "cape" => "Phần sau lưng đung đưa (áo choàng…) — tô cam",
        "skirt" => "Vạt dưới hông đung đưa (váy, vạt áo) — tô xanh",
        "hair" => "Tóc dài đung đưa — tô tím",
        _ => Name,
    };

    public bool On
    {
        get => _on;
        set { if (Set(ref _on, value)) _ = _changed(this); }
    }

    /// <summary>The backend's value, without sending it back.</summary>
    public void Load(bool on) => Set(ref _on, on, nameof(On));
}

/// <summary>
/// The whole workflow of §11.2: choose the character, draw its rig, choose the animation (and whether the hands
/// grip), generate, watch progress, preview, regenerate a finished frame or all and compare, export.
/// </summary>
public sealed class MainViewModel : ObservableObject
{
    private BackendClient? _api;
    private readonly DispatcherTimer _player;
    private string? _taskId;

    public MainViewModel()
    {
        _player = new DispatcherTimer(DispatcherPriority.Render);
        _player.Tick += (_, _) => NextPlayFrame();
        for (int i = 1; i <= 8; i++) Frames.Add(new FrameItem(i));

        ChooseCharacterCommand = new AsyncCommand(_ => ChooseCharacterAsync(), OnError, _ => Ready && !IsBusy);
        FlipFacingCommand = new AsyncCommand(_ => FlipFacingAsync(), OnError,
                                             _ => Ready && !IsBusy && _characterPath != null);
        MakeRigCommand = new AsyncCommand(_ => MakeRigAsync(null), OnError,
                                          _ => Ready && !IsBusy && _skeleton is { Confirmed: true });
        RedrawRigPartCommand = new AsyncCommand(p => MakeRigAsync(p as string), OnError, _ => Ready && !IsBusy && HasRig);
        EditRigCommand = new AsyncCommand(_ => EditRigAsync(), OnError, _ => Ready && !IsBusy && HasPartsFile);
        ReloadRigCommand = new AsyncCommand(_ => ReloadRigAsync(), OnError, _ => Ready && !IsBusy && IsEditingRig);
        GenerateCommand = new AsyncCommand(_ => GenerateAsync(), OnError, _ => CanGenerate);
        RegenerateAllCommand = new AsyncCommand(_ => RegenerateAsync(null), OnError,
                                                _ => Ready && !IsBusy && HasFrames && CanRedraw);
        RegenerateFrameCommand = new AsyncCommand(_ => RegenerateAsync(SelectedFrame?.Index), OnError,
                                                  _ => Ready && !IsBusy && HasFrames && CanRedraw && SelectedFrame != null);
        AcceptCommand = new AsyncCommand(_ => AcceptAsync(true), OnError,
                                         _ => !IsBusy && SelectedFrame is { HasCandidate: true });
        KeepCommand = new AsyncCommand(_ => AcceptAsync(false), OnError,
                                       _ => !IsBusy && SelectedFrame is { HasCandidate: true } or { CandidateFailed: true });
        AcceptAllCommand = new AsyncCommand(_ => AcceptAllAsync(), OnError,
                                            _ => !IsBusy && Frames.Any(f => f.HasCandidate));
        CancelCommand = new AsyncCommand(_ => CancelAsync(), OnError, _ => IsBusy && _taskId != null);
        ExportCommand = new AsyncCommand(_ => ExportAsync(), OnError, _ => Ready && !IsBusy && _complete);
        BrowseExportCommand = new RelayCommand(_ => BrowseExport(), _ => !IsBusy);
        OpenExportFolderCommand = new RelayCommand(_ => OpenFolder(ExportFolder), _ => Directory.Exists(ExportFolder));
        PlayPauseCommand = new RelayCommand(_ => TogglePlay(), _ => HasFrames);
        SelectFrameCommand = new RelayCommand(p => SelectFrame(p as FrameItem));

        ExportFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "AnimGen");
        SetStatus(StatusKind.Busy, "Đang khởi động phần xử lý…");
    }

    // ---------------------------------------------------------------------------------------------------------
    // Bindable state
    // ---------------------------------------------------------------------------------------------------------
    public ObservableCollection<AnimationInfo> Animations { get; } = new();
    public ObservableCollection<FrameItem> Frames { get; } = new();
    public IReadOnlyList<int> FpsChoices { get; } = new[] { 6, 8, 12, 24 };          // cut C7
    public ObservableCollection<int> ExportHeights { get; } = new() { 128, 256, 512 };

    private bool _ready;
    public bool Ready { get => _ready; private set => Set(ref _ready, value); }

    private bool _isDemo;
    public bool IsDemo { get => _isDemo; private set => Set(ref _isDemo, value); }

    private bool _isBusy;
    public bool IsBusy { get => _isBusy; private set { if (Set(ref _isBusy, value)) Raise(nameof(CanGenerate)); } }

    private StatusKind _statusKind = StatusKind.Info;
    public StatusKind StatusKind { get => _statusKind; private set => Set(ref _statusKind, value); }

    private string _statusText = "";
    public string StatusText { get => _statusText; private set => Set(ref _statusText, value); }

    private string _progressText = "";
    public string ProgressText { get => _progressText; private set => Set(ref _progressText, value); }

    private double _progressValue;
    public double ProgressValue { get => _progressValue; private set => Set(ref _progressValue, value); }

    private double _progressMax = 8;
    public double ProgressMax { get => _progressMax; private set => Set(ref _progressMax, value); }

    private string? _characterPath;
    private ImageSource? _characterImage;
    public ImageSource? CharacterImage { get => _characterImage; private set => Set(ref _characterImage, value); }

    private string _characterText = "Chưa chọn ảnh nhân vật.";
    public string CharacterText { get => _characterText; private set => Set(ref _characterText, value); }

    private SkeletonInfo? _skeleton;
    public SkeletonInfo? Skeleton { get => _skeleton; private set { if (Set(ref _skeleton, value)) { Raise(nameof(SkeletonText)); Raise(nameof(HasSkeleton)); Raise(nameof(CanGenerate)); } } }
    public bool HasSkeleton => _skeleton != null;
    public string SkeletonText => _skeleton == null ? "" : _skeleton.Confirmed
        ? "Khung xương: đã xác nhận."
        : "Khung xương: chưa xác nhận — bấm “Kiểm tra” trước khi tạo.";

    /// <summary>Raised after a character loads with a skeleton the owner has not confirmed yet.</summary>
    public event EventHandler? SkeletonCheckRequested;

    // Step 2: the character's rig -------------------------------------------------------------------------------
    private ImageSource? _rigImage;
    public ImageSource? RigImage { get => _rigImage; private set => Set(ref _rigImage, value); }

    private ImageSource? _handImage;
    public ImageSource? HandImage { get => _handImage; private set => Set(ref _handImage, value); }

    private bool _hasRig;
    public bool HasRig
    {
        get => _hasRig;
        private set { if (Set(ref _hasRig, value)) { Raise(nameof(RigText)); Raise(nameof(RigButtonText)); Raise(nameof(CanGenerate)); } }
    }

    public string RigText => HasRig
        ? "Phần tô màu sẽ đung đưa khi chuyển động. Bộ phận nào sai thì bấm “Sửa trong Photoshop”: mỗi bộ phận là một "
          + "lớp riêng; sửa, lưu lại (Ctrl+S) là khung nhân vật được dựng lại. Hình nào hỏng hẳn thì bấm “Vẽ lại”."
        : "AI vẽ nhân vật ở tư thế dang tay chân và một bàn tay mở, rồi tách từng bộ phận và vẽ nốt phần bị che "
          + "(khoảng 8 phút). Mọi khung hình được làm từ các bộ phận này.";

    public string RigButtonText => HasRig ? "Vẽ lại cả hai hình" : "Vẽ khung nhân vật";

    private bool _isEditingRig;
    public bool IsEditingRig { get => _isEditingRig; private set => Set(ref _isEditingRig, value); }

    private bool _hasPartsFile;
    public bool HasPartsFile { get => _hasPartsFile; private set => Set(ref _hasPartsFile, value); }

    public ObservableCollection<HangingItem> HangingParts { get; } = new();

    private bool _hasHanging;
    public bool HasHanging { get => _hasHanging; private set => Set(ref _hasHanging, value); }

    /// <summary>The AI finishing pass (owner, 2026-10-30: optional, off by default).</summary>
    private bool _finish;
    public bool FinishWithAi { get => _finish; set => Set(ref _finish, value); }

    /// <summary>Only finished frames can be redrawn: frames made from the rig come out the same every time.</summary>
    private bool _canRedraw;
    public bool CanRedraw { get => _canRedraw; private set => Set(ref _canRedraw, value); }

    private string _facingText = "";
    public string FacingText { get => _facingText; private set => Set(ref _facingText, value); }
    private string _facing = "right";

    private AnimationInfo? _selectedAnimation;
    public AnimationInfo? SelectedAnimation
    {
        get => _selectedAnimation;
        set
        {
            if (Set(ref _selectedAnimation, value))
            {
                Raise(nameof(ShowGrip));
                Raise(nameof(CanGenerate));
                Raise(nameof(AnimationNote));
            }
        }
    }

    /// <summary>The app draws characters only; weapons are attached in the game (owner, 2026-10-30). Where the
    /// animation leaves it open, the owner chooses whether the hands are drawn closed around a weapon.</summary>
    public bool ShowGrip => SelectedAnimation is { Ready: true, Grip: "optional" };

    private bool _grip;
    public bool Grip { get => _grip; set => Set(ref _grip, value); }

    public string AnimationNote => SelectedAnimation switch
    {
        null => "",
        { Ready: false } => "Hoạt ảnh này chưa có trong phiên bản này.",
        { Grip: "always" } => "Tay nắm như đang cầm vũ khí. Khi xuất có kèm tệp vị trí tay (…_weapon.json) để gắn vũ khí trong game.",
        { Grip: "optional" } => "Nếu chọn tay nắm, khi xuất có kèm tệp vị trí tay (…_weapon.json) để gắn vũ khí trong game.",
        _ => "Tay không, không có vị trí gắn vũ khí.",
    };

    private FrameItem? _selectedFrame;
    public FrameItem? SelectedFrame
    {
        get => _selectedFrame;
        private set
        {
            if (_selectedFrame != null) _selectedFrame.IsSelected = false;
            if (Set(ref _selectedFrame, value))
            {
                if (value != null) value.IsSelected = true;
                Raise(nameof(ShowCompare));
            }
        }
    }

    public bool ShowCompare => SelectedFrame is { HasCandidate: true } or { CandidateFailed: true };

    private ImageSource? _previewImage;
    public ImageSource? PreviewImage { get => _previewImage; private set => Set(ref _previewImage, value); }

    private bool _isPlaying;
    public bool IsPlaying { get => _isPlaying; private set => Set(ref _isPlaying, value); }

    private int _fps = 8;
    public int Fps
    {
        get => _fps;
        set
        {
            if (Set(ref _fps, value)) _player.Interval = TimeSpan.FromMilliseconds(1000.0 / Math.Max(1, value));
        }
    }

    private int _exportHeight = 256;
    public int ExportHeight { get => _exportHeight; set => Set(ref _exportHeight, value); }

    private string _exportFolder = "";
    public string ExportFolder { get => _exportFolder; set => Set(ref _exportFolder, value); }

    private bool _hasFrames;
    public bool HasFrames { get => _hasFrames; private set => Set(ref _hasFrames, value); }

    private bool _complete;

    private string _framesSummary = "";
    public string FramesSummary { get => _framesSummary; private set => Set(ref _framesSummary, value); }

    public bool CanGenerate =>
        Ready && !IsBusy && _characterPath != null && _skeleton is { Confirmed: true } && HasRig
        && SelectedAnimation is { Ready: true };

    public ICommand ChooseCharacterCommand { get; }
    public ICommand FlipFacingCommand { get; }
    public ICommand MakeRigCommand { get; }
    public ICommand RedrawRigPartCommand { get; }
    public ICommand EditRigCommand { get; }
    public ICommand ReloadRigCommand { get; }
    public ICommand GenerateCommand { get; }
    public ICommand RegenerateAllCommand { get; }
    public ICommand RegenerateFrameCommand { get; }
    public ICommand AcceptCommand { get; }
    public ICommand KeepCommand { get; }
    public ICommand AcceptAllCommand { get; }
    public ICommand CancelCommand { get; }
    public ICommand ExportCommand { get; }
    public ICommand BrowseExportCommand { get; }
    public ICommand OpenExportFolderCommand { get; }
    public ICommand PlayPauseCommand { get; }
    public ICommand SelectFrameCommand { get; }

    public BackendClient? Api => _api;

    // ---------------------------------------------------------------------------------------------------------
    // Start-up
    // ---------------------------------------------------------------------------------------------------------
    public async Task InitializeAsync(BackendClient api)
    {
        _api = api;
        HealthReply h = await api.HealthAsync();
        IsDemo = h.Demo;
        Animations.Clear();
        foreach (AnimationInfo a in await api.AnimationsAsync()) Animations.Add(a);
        SelectedAnimation = Animations.FirstOrDefault(a => a.Ready);
        Fps = 8;
        Ready = true;
        SetStatus(StatusKind.Info, "Sẵn sàng. Hãy chọn ảnh nhân vật.");
        CommandManager.InvalidateRequerySuggested();
    }

    public void ShowBusy(string message) => SetStatus(StatusKind.Busy, message);

    public void ShowStartupError(string message)
    {
        Ready = false;
        SetStatus(StatusKind.Error, message);
    }

    // ---------------------------------------------------------------------------------------------------------
    // Step 1: character
    // ---------------------------------------------------------------------------------------------------------
    private async Task ChooseCharacterAsync()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Chọn ảnh nhân vật (nền trong suốt)",
            Filter = "Ảnh PNG hoặc WebP|*.png;*.webp|Mọi tệp|*.*",
        };
        if (dlg.ShowDialog() != true) return;
        await SetCharacterAsync(dlg.FileName, null);
    }

    private async Task FlipFacingAsync()
    {
        if (_characterPath == null) return;
        await SetCharacterAsync(_characterPath, _facing == "left" ? "right" : "left");
    }

    private async Task SetCharacterAsync(string path, string? facing)
    {
        if (_api == null) return;
        SetStatus(StatusKind.Busy, "Đang kiểm tra ảnh…");
        CharacterReply r;
        try
        {
            r = await _api.SetCharacterAsync(path, facing, Path.GetFileNameWithoutExtension(path));
        }
        catch (BackendException ex)
        {
            // An invalid input is never treated as valid (§4.4): the previous character stays, if any.
            SetStatus(StatusKind.Error, $"Ảnh không dùng được: {ex.Message}");
            return;
        }
        _characterPath = path;
        CharacterImage = FrameItem.Load(path);
        CharacterInfo c = r.Character!;
        CharacterText = $"{Path.GetFileName(path)} — {c.Width}×{c.Height} px, nhân vật cao {c.FigureHeight} px.";
        _facing = c.Facing;
        string dir = c.Facing == "left" ? "quay trái" : "quay phải";
        FacingText = c.FacingSource switch
        {
            "detected" => $"Hướng: {dir} (tự nhận ra).",
            "owner" => $"Hướng: {dir} (bạn chọn).",
            "saved" => $"Hướng: {dir} (đã lưu cùng khung xương).",
            _ => $"Hướng: {dir} (giả định — bấm “Đổi hướng” nếu sai).",
        };
        Skeleton = c.Skeleton;
        if (r.State != null) ApplyState(r.State);
        CommandManager.InvalidateRequerySuggested();
        if (c.Skeleton is { Confirmed: false })
        {
            SetStatus(StatusKind.Info, "Ảnh hợp lệ. Hãy kiểm tra khung xương của nhân vật.");
            SkeletonCheckRequested?.Invoke(this, EventArgs.Empty);
        }
        else
        {
            SetStatus(StatusKind.Ok, HasRig ? "Ảnh hợp lệ. Hãy chọn hoạt ảnh rồi bấm Tạo."
                                            : "Ảnh hợp lệ. Hãy vẽ khung nhân vật (bước 2).");
        }
    }

    /// <summary>The skeleton dialog's reply: the confirmed skeleton, and the frames the backend dropped with the old one.</summary>
    public void ApplySkeletonReply(CharacterReply r)
    {
        Skeleton = r.Character?.Skeleton;
        if (r.State != null) ApplyState(r.State);
        SetStatus(StatusKind.Ok, HasRig ? "Đã xác nhận khung xương. Hãy chọn hoạt ảnh rồi bấm Tạo."
                                        : "Đã xác nhận khung xương. Hãy vẽ khung nhân vật (bước 2).");
        CommandManager.InvalidateRequerySuggested();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Step 2: rig
    // ---------------------------------------------------------------------------------------------------------
    /// <param name="part">"rig" or "hand": only that drawing; null: both.</param>
    private async Task MakeRigAsync(string? part)
    {
        if (_api == null) return;
        // A new spread-pose drawing makes every part again; only the hand keeps the owner's corrections.
        if (IsEditingRig && part != "hand"
            && System.Windows.MessageBox.Show(System.Windows.Application.Current.MainWindow,
                   "Vẽ lại sẽ làm lại mọi bộ phận, những chỗ bạn đã sửa trong Photoshop sẽ mất. Vẫn vẽ lại?",
                   "Vẽ lại khung nhân vật", System.Windows.MessageBoxButton.YesNo,
                   System.Windows.MessageBoxImage.Warning) != System.Windows.MessageBoxResult.Yes)
            return;
        StopPlay();
        TaskReply r = await _api.MakeRigAsync(null, part);
        await RunTaskAsync(r.Task!);
    }

    /// <summary>The body parts open in Photoshop, one layer each (owner, 2026-10-31); saving rebuilds the rig.</summary>
    private async Task EditRigAsync()
    {
        if (_api == null) return;
        EditReply r = await _api.EditRigAsync();
        if (r.State != null) ApplyState(r.State);
        WatchPartsFile(r.File);
        string? photoshop = PhotoshopPath();
        try
        {
            Process.Start(photoshop != null
                ? new ProcessStartInfo(photoshop, $"\"{r.File}\"") { UseShellExecute = false }
                : new ProcessStartInfo(r.File) { UseShellExecute = true });
        }
        catch (Exception)
        {
            OpenFolder(Path.GetDirectoryName(r.File)!);
        }
        SetStatus(StatusKind.Ok, photoshop != null
            ? "Đã mở các bộ phận trong Photoshop. Sửa từng lớp (giữ tên lớp, đừng đổi cỡ hình), rồi lưu lại (Ctrl+S)."
            : "Không thấy Photoshop nên đã mở tệp bằng chương trình mặc định. Sửa từng lớp rồi lưu lại dạng PSD.");
    }

    /// <summary>Photoshop's own path, as Windows registers it for "Run" (App Paths); null if not installed.</summary>
    private static string? PhotoshopPath()
    {
        foreach (RegistryKey root in new[] { Registry.LocalMachine, Registry.CurrentUser })
        {
            try
            {
                using RegistryKey? key = root.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe");
                if (key?.GetValue(null) is string path && File.Exists(path.Trim('"'))) return path.Trim('"');
            }
            catch (Exception) { }
        }
        return null;
    }

    // Photoshop saves through a temporary file and a rename, in several writes: the rebuild waits until it is quiet.
    private FileSystemWatcher? _partsWatcher;
    private DispatcherTimer? _partsQuiet;

    private void WatchPartsFile(string file)
    {
        if (_partsWatcher != null && _partsWatcher.Path == Path.GetDirectoryName(file)) return;
        StopWatchingParts();
        _partsQuiet = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1.5) };
        _partsQuiet.Tick += async (_, _) =>
        {
            if (IsBusy) return;                         // tried again on the next tick
            _partsQuiet!.Stop();
            await ReloadRigAsync(fromSave: true);
        };
        _partsWatcher = new FileSystemWatcher(Path.GetDirectoryName(file)!, Path.GetFileName(file))
        {
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.Size,
        };
        FileSystemEventHandler changed = (_, _) => System.Windows.Application.Current.Dispatcher.BeginInvoke(() =>
        {
            _partsQuiet?.Stop();
            _partsQuiet?.Start();
        });
        _partsWatcher.Changed += changed;
        _partsWatcher.Created += changed;
        _partsWatcher.Renamed += (s, e) => changed(s, e);
        _partsWatcher.EnableRaisingEvents = true;
    }

    private void StopWatchingParts()
    {
        _partsQuiet?.Stop();
        _partsQuiet = null;
        _partsWatcher?.Dispose();
        _partsWatcher = null;
    }

    private Task ReloadRigAsync() => ReloadRigAsync(fromSave: false);

    private async Task ReloadRigAsync(bool fromSave)
    {
        if (_api == null) return;
        try
        {
            StopPlay();
            ApplyState(await _api.ReloadRigAsync());
            SetStatus(StatusKind.Ok, "Đã dựng lại khung nhân vật từ các bộ phận đã sửa.");
        }
        catch (BackendException ex) when (fromSave)
        {
            SetStatus(StatusKind.Error, ex.Message);    // saved from Photoshop: told in the status line, no dialog
        }
    }

    private async Task HangingChangedAsync(HangingItem item)
    {
        if (_api == null) return;
        try
        {
            ApplyState(await _api.SetHangingAsync(item.Name, item.On));
            SetStatus(StatusKind.Ok, item.On ? "Phần này sẽ đung đưa." : "Phần này sẽ đứng yên như hình vẽ.");
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
        }
    }

    private void ApplyRig(RigInfo? rig)
    {
        HasRig = rig != null;
        RigImage = rig?.View != null ? FrameItem.Load(rig.View) : null;
        HandImage = rig?.HandView != null ? FrameItem.Load(rig.HandView) : null;
        var parts = rig?.Hanging ?? new List<HangingInfo>();
        if (!parts.Select(p => p.Name).SequenceEqual(HangingParts.Select(h => h.Name)))
        {
            HangingParts.Clear();
            foreach (HangingInfo p in parts) HangingParts.Add(new HangingItem(p.Name, p.On, HangingChangedAsync));
        }
        else
        {
            for (int i = 0; i < parts.Count; i++) HangingParts[i].Load(parts[i].On);
        }
        HasHanging = HangingParts.Count > 0;
        IsEditingRig = rig is { Editing.Count: > 0 };
        HasPartsFile = rig?.Psd != null;
        if (!IsEditingRig) StopWatchingParts();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Generate / regenerate / compare
    // ---------------------------------------------------------------------------------------------------------
    private async Task GenerateAsync()
    {
        if (_api == null || SelectedAnimation == null) return;
        StopPlay();
        TaskReply r = await _api.GenerateAsync(SelectedAnimation.Id, ShowGrip ? Grip : null, FinishWithAi);
        await RunTaskAsync(r.Task!);
    }

    private async Task RegenerateAsync(int? frame)
    {
        if (_api == null) return;
        StopPlay();
        TaskReply r = await _api.RegenerateAsync(frame);
        await RunTaskAsync(r.Task!);
        if (frame != null) SelectFrame(Frames[frame.Value - 1]);
    }

    private async Task AcceptAsync(bool accept)
    {
        if (_api == null || SelectedFrame == null) return;
        int i = SelectedFrame.Index;
        SessionState s = accept ? await _api.AcceptAsync(i) : await _api.KeepAsync(i);
        ApplyState(s);
        SetStatus(StatusKind.Ok, accept ? $"Đã dùng bản mới cho khung {i}." : $"Đã giữ bản cũ của khung {i}.");
        PreviewImage = SelectedFrame.Image;
    }

    private async Task AcceptAllAsync()
    {
        if (_api == null) return;
        ApplyState(await _api.AcceptAllAsync());
        SetStatus(StatusKind.Ok, "Đã dùng tất cả bản mới.");
    }

    private async Task CancelAsync()
    {
        if (_api == null || _taskId == null) return;
        await _api.CancelAsync(_taskId);
        ProgressText = "Đang huỷ sau bước hiện tại…";
    }

    private async Task RunTaskAsync(TaskInfo task)
    {
        if (_api == null) return;
        _taskId = task.Id;
        IsBusy = true;
        CommandManager.InvalidateRequerySuggested();
        int lastDone = -1;
        try
        {
            while (true)
            {
                ShowProgress(task);
                if (task.Done != lastDone)
                {
                    lastDone = task.Done;
                    ApplyState(await _api.StateAsync());
                }
                if (task.Finished) break;
                await Task.Delay(300);
                task = await _api.TaskAsync(task.Id);
            }
            ApplyState(await _api.StateAsync());
            Finish(task);
        }
        finally
        {
            _taskId = null;
            IsBusy = false;
            ProgressText = "";
            CommandManager.InvalidateRequerySuggested();
        }
    }

    private void ShowProgress(TaskInfo t)
    {
        ProgressMax = Math.Max(1, t.Total);
        ProgressValue = t.Done;
        switch (t.Stage)
        {
            case "loading_models":
                SetStatus(StatusKind.Busy, "Đang tải mô hình… (lần tạo đầu tiên mất thêm 1–2 phút)");
                ProgressText = t.Message;
                break;
            case "generating" when t.Kind == "rig":
                SetStatus(StatusKind.Busy, "Đang vẽ khung nhân vật…");
                ProgressText = t.Message;
                break;
            case "generating":
                SetStatus(StatusKind.Busy, "Đang tạo…");
                ProgressText = $"Khung {t.Frame} / 8";
                break;
            case "retrying":
                SetStatus(StatusKind.Busy, "Đang thử lại khung lỗi…");
                ProgressText = $"Khung {t.Frame} / 8 — thử lại lần {t.Attempt}";
                break;
            case "exporting":
                SetStatus(StatusKind.Busy, "Đang xuất…");
                ProgressText = $"{t.Done} / {t.Total}";
                break;
        }
    }

    private void Finish(TaskInfo t)
    {
        bool export = t.Kind == "export";
        switch (t.State)
        {
            case "completed" when t.Kind == "rig":
                SetStatus(StatusKind.Ok, "Đã vẽ khung nhân vật. Kiểm tra hình ở bước 2, rồi chọn hoạt ảnh và bấm Tạo.");
                return;
            case "failed" when t.Kind == "rig":
                SetStatus(StatusKind.Error, $"Vẽ khung nhân vật thất bại: {t.Error?.Message ?? "Lỗi không rõ."}");
                return;
            case "completed" when export:
                SetStatus(StatusKind.Ok, $"Đã xuất xong 8 khung và sprite sheet vào: {ExportFolder}");
                break;
            case "completed":
                var failed = Frames.Where(f => f.IsFailed || f.CandidateFailed).Select(f => f.Index).ToList();
                if (failed.Count > 0)
                    SetStatus(StatusKind.Warning,
                              $"Xong, nhưng khung {string.Join(", ", failed)} vẫn lỗi sau khi tự thử lại. "
                              + "Chọn khung đó rồi bấm “Tạo lại khung này”.");
                else if (Frames.Any(f => f.HasCandidate))
                    SetStatus(StatusKind.Ok, "Đã tạo bản mới. So sánh rồi chọn “Dùng bản mới” hoặc “Giữ bản cũ”.");
                else
                    SetStatus(StatusKind.Ok, $"Đã tạo xong 8 khung ({t.Elapsed:0} giây).");
                if (!export && HasFrames) StartPlay();
                break;
            case "cancelled":
                SetStatus(StatusKind.Info, "Đã huỷ.");
                break;
            default:
                string msg = t.Error?.Message ?? "Lỗi không rõ.";
                SetStatus(StatusKind.Error, export ? $"Xuất thất bại: {msg}" : $"Tạo thất bại: {msg}");
                break;
        }
    }

    private SessionState? _lastState;

    private void ApplyState(SessionState s)
    {
        _lastState = s;
        IsDemo = s.Demo;
        ApplyRig(s.Rig);
        CanRedraw = s.Finished;
        for (int i = 0; i < Frames.Count && i < s.Frames.Count; i++) Frames[i].Update(s.Frames[i]);
        HasFrames = Frames.Any(f => f.Image != null);
        _complete = s.Complete;
        if (s.ExportHeights.Count > 0 && !s.ExportHeights.SequenceEqual(ExportHeights))
        {
            ExportHeights.Clear();
            foreach (int h in s.ExportHeights) ExportHeights.Add(h);
        }
        int ok = Frames.Count(f => f.IsOk);
        FramesSummary = s.Animation == null ? "" : $"{s.Animation.NameVi}: {ok} / 8 khung xong.";
        if (!IsPlaying) PreviewImage = SelectedFrame?.Image ?? Frames.FirstOrDefault(f => f.Image != null)?.Image;
        Raise(nameof(ShowCompare));
        CommandManager.InvalidateRequerySuggested();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Preview
    // ---------------------------------------------------------------------------------------------------------
    private int _playIndex;

    private void TogglePlay()
    {
        if (IsPlaying) StopPlay();
        else StartPlay();
    }

    private void StartPlay()
    {
        SelectedFrame = null;
        _player.Interval = TimeSpan.FromMilliseconds(1000.0 / Math.Max(1, Fps));
        IsPlaying = true;
        _player.Start();
    }

    private void StopPlay()
    {
        _player.Stop();
        IsPlaying = false;
    }

    private void NextPlayFrame()
    {
        var shown = Frames.Where(f => f.Image != null).ToList();
        if (shown.Count == 0) return;
        _playIndex = (_playIndex + 1) % shown.Count;
        PreviewImage = shown[_playIndex].Image;
    }

    private void SelectFrame(FrameItem? f)
    {
        StopPlay();
        SelectedFrame = f;
        PreviewImage = f?.Image;
        CommandManager.InvalidateRequerySuggested();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Export
    // ---------------------------------------------------------------------------------------------------------
    private void BrowseExport()
    {
        var dlg = new OpenFolderDialog { Title = "Chọn thư mục xuất" };
        if (Directory.Exists(ExportFolder)) dlg.InitialDirectory = ExportFolder;
        if (dlg.ShowDialog() == true) ExportFolder = dlg.FolderName;
    }

    private async Task ExportAsync()
    {
        if (_api == null) return;
        if (Frames.Any(f => f.HasCandidate))
        {
            SetStatus(StatusKind.Warning, "Còn khung có bản mới chưa chọn. Hãy “Dùng bản mới” hoặc “Giữ bản cũ” trước khi xuất.");
            return;
        }
        TaskReply r = await _api.ExportAsync(ExportFolder, ExportHeight);
        await RunTaskAsync(r.Task!);
    }

    private static void OpenFolder(string folder)
    {
        try { Process.Start(new ProcessStartInfo("explorer.exe", $"\"{folder}\"") { UseShellExecute = true }); }
        catch (Exception) { /* nothing to do */ }
    }

    // ---------------------------------------------------------------------------------------------------------
    private void SetStatus(StatusKind kind, string text)
    {
        StatusKind = kind;
        StatusText = text;
    }

    private void OnError(Exception ex)
    {
        string msg = ex is BackendException b ? b.Message : $"Lỗi: {ex.Message}";
        SetStatus(StatusKind.Error, msg);
    }
}
