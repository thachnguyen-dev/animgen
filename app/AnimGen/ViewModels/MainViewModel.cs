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

public sealed class WeaponOption
{
    public WeaponInfo? Weapon { get; init; }
    public string Label => Weapon == null ? "Không dùng vũ khí" : Weapon.ToString();
    public override string ToString() => Label;
}

/// <summary>
/// The whole workflow of §11.2: choose the character, choose the animation (and variant, weapon, skin),
/// generate, watch progress, preview, regenerate a frame or all and compare, export.
/// </summary>
public sealed class MainViewModel : ObservableObject
{
    private BackendClient? _api;
    private readonly DispatcherTimer _player;
    private bool _suppressWeaponChange;
    private string? _taskId;

    public MainViewModel()
    {
        _player = new DispatcherTimer(DispatcherPriority.Render);
        _player.Tick += (_, _) => NextPlayFrame();
        for (int i = 1; i <= 8; i++) Frames.Add(new FrameItem(i));

        ChooseCharacterCommand = new AsyncCommand(_ => ChooseCharacterAsync(), OnError, _ => Ready && !IsBusy);
        FlipFacingCommand = new AsyncCommand(_ => FlipFacingAsync(), OnError,
                                             _ => Ready && !IsBusy && _characterPath != null);
        GenerateCommand = new AsyncCommand(_ => GenerateAsync(), OnError, _ => CanGenerate);
        RegenerateAllCommand = new AsyncCommand(_ => RegenerateAsync(null), OnError,
                                                _ => Ready && !IsBusy && HasFrames);
        RegenerateFrameCommand = new AsyncCommand(_ => RegenerateAsync(SelectedFrame?.Index), OnError,
                                                  _ => Ready && !IsBusy && HasFrames && SelectedFrame != null);
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
    public ObservableCollection<WeaponOption> WeaponOptions { get; } = new();
    public ObservableCollection<SkinInfo> Skins { get; } = new();
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
                RebuildWeaponOptions();
                Raise(nameof(ShowWeapon));
                Raise(nameof(CanGenerate));
                Raise(nameof(AnimationNote));
            }
        }
    }

    public bool ShowWeapon => SelectedAnimation is { Weapon: "optional" or "required" };

    public string AnimationNote => SelectedAnimation switch
    {
        null => "",
        { Ready: false } => "Hoạt ảnh này chưa có trong phiên bản này.",
        { Weapon: "required" } => "Cần chọn vũ khí phù hợp.",
        { Weapon: "optional" } => "Vũ khí là tuỳ chọn. Khi xuất có kèm tệp vị trí tay cầm (…_weapon.json) để gắn vũ khí trong game.",
        _ => "Không dùng vũ khí.",
    };

    private WeaponOption? _selectedWeapon;
    public WeaponOption? SelectedWeapon
    {
        get => _selectedWeapon;
        set
        {
            if (Set(ref _selectedWeapon, value))
            {
                RebuildSkins();
                Raise(nameof(CanGenerate));
                _ = RecombineIfNeededAsync();
            }
        }
    }

    private SkinInfo? _selectedSkin;
    public SkinInfo? SelectedSkin
    {
        get => _selectedSkin;
        set
        {
            if (Set(ref _selectedSkin, value))
            {
                Raise(nameof(CanGenerate));
                _ = RecombineIfNeededAsync();
            }
        }
    }

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
        Ready && !IsBusy && _characterPath != null && _skeleton is { Confirmed: true } && SelectedAnimation is { Ready: true }
        && (SelectedAnimation.Weapon != "required"
            || (SelectedWeapon?.Weapon != null && SelectedSkin is { Ready: true }))
        && (SelectedWeapon?.Weapon == null || SelectedSkin is { Ready: true });

    public ICommand ChooseCharacterCommand { get; }
    public ICommand FlipFacingCommand { get; }
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
        await ReloadWeaponsAsync();
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

    public async Task ReloadWeaponsAsync(string? selectWeaponId = null)
    {
        if (_api == null) return;
        _weapons = await _api.WeaponsAsync();
        RebuildWeaponOptions(selectWeaponId);
    }

    private List<WeaponInfo> _weapons = new();

    private void RebuildWeaponOptions(string? selectId = null)
    {
        _suppressWeaponChange = true;
        try
        {
            string? keep = selectId ?? SelectedWeapon?.Weapon?.Id;
            WeaponOptions.Clear();
            AnimationInfo? a = SelectedAnimation;
            if (a == null || a.Weapon == "never")
            {
                SelectedWeapon = null;
                return;
            }
            if (a.Weapon == "optional") WeaponOptions.Add(new WeaponOption());
            foreach (WeaponInfo w in _weapons.Where(w => a.WeaponKinds.Contains(w.Kind)))
                WeaponOptions.Add(new WeaponOption { Weapon = w });
            SelectedWeapon = WeaponOptions.FirstOrDefault(o => o.Weapon?.Id == keep) ?? WeaponOptions.FirstOrDefault();
            RebuildSkins();
        }
        finally
        {
            _suppressWeaponChange = false;
        }
    }

    private void RebuildSkins()
    {
        bool outer = _suppressWeaponChange;
        _suppressWeaponChange = true;
        try
        {
            string? keep = SelectedSkin?.Name;
            Skins.Clear();
            if (SelectedWeapon?.Weapon != null)
                foreach (SkinInfo s in SelectedWeapon.Weapon.Skins) Skins.Add(s);
            SelectedSkin = Skins.FirstOrDefault(s => s.Name == keep) ?? Skins.FirstOrDefault();
        }
        finally
        {
            _suppressWeaponChange = outer;
        }
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
            SetStatus(StatusKind.Ok, "Ảnh hợp lệ. Hãy chọn hoạt ảnh rồi bấm Tạo.");
        }
    }

    /// <summary>The skeleton dialog's reply: the confirmed skeleton, and the frames the backend dropped with the old one.</summary>
    public void ApplySkeletonReply(CharacterReply r)
    {
        Skeleton = r.Character?.Skeleton;
        if (r.State != null) ApplyState(r.State);
        SetStatus(StatusKind.Ok, "Đã xác nhận khung xương. Hãy chọn hoạt ảnh rồi bấm Tạo.");
        CommandManager.InvalidateRequerySuggested();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Generate / regenerate / compare
    // ---------------------------------------------------------------------------------------------------------
    private WeaponRef? CurrentWeapon() =>
        SelectedWeapon?.Weapon == null ? null : new WeaponRef { WeaponId = SelectedWeapon.Weapon.Id, Skin = SelectedSkin?.Name };

    private async Task GenerateAsync()
    {
        if (_api == null || SelectedAnimation == null) return;
        StopPlay();
        TaskReply r = await _api.GenerateAsync(SelectedAnimation.Id, CurrentWeapon());
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

    private async Task RecombineIfNeededAsync()
    {
        // Changing weapon or skin after generation only recombines the frames (§7): seconds, no AI work.
        if (_suppressWeaponChange || _api == null || IsBusy || !HasFrames || _lastState?.Animation == null) return;
        if (SelectedAnimation?.Id != _lastState.Animation.Id) return;
        WeaponRef? w = CurrentWeapon();
        if (w != null && SelectedSkin is not { Ready: true }) return;
        if (_lastState.Animation.Weapon == "required" && w == null) return;
        try
        {
            ApplyState(await _api.SetWeaponAsync(w));
            SetStatus(StatusKind.Ok, w == null ? "Đã bỏ vũ khí khỏi các khung." : "Đã ghép lại vũ khí vào các khung.");
        }
        catch (Exception ex)
        {
            OnError(ex);
        }
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
        for (int i = 0; i < Frames.Count && i < s.Frames.Count; i++) Frames[i].Update(s.Frames[i]);
        HasFrames = Frames.Any(f => f.Image != null);
        _complete = s.Complete;
        if (s.ExportHeights.Count > 0 && !s.ExportHeights.SequenceEqual(ExportHeights))
        {
            ExportHeights.Clear();
            foreach (int h in s.ExportHeights) ExportHeights.Add(h);
        }
        int ok = Frames.Count(f => f.IsOk);
        var clipped = Frames.Where(f => f.WeaponClipped).Select(f => f.Index).ToList();
        FramesSummary = s.Animation == null
            ? ""
            : $"{s.Animation.NameVi}: {ok} / 8 khung xong."
              + (clipped.Count > 0 ? $" Vũ khí bị cắt ở mép khung: {string.Join(", ", clipped)}." : "");
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
