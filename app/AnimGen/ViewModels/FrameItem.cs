using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using AnimGen.Services;

namespace AnimGen.ViewModels;

/// <summary>One of the 8 frames in the strip, with its regenerated candidate when there is one.</summary>
public sealed class FrameItem : ObservableObject
{
    private ImageSource? _image;
    private ImageSource? _candidate;
    private string _status = "empty";
    private string _candidateStatus = "none";
    private string _errorText = "";
    private bool _weaponClipped;
    private bool _isSelected;
    private string? _previewPath;
    private string? _candidatePath;

    public FrameItem(int index) => Index = index;

    public int Index { get; }
    public string Label => $"Khung {Index}";

    public ImageSource? Image { get => _image; private set => Set(ref _image, value); }
    public ImageSource? Candidate { get => _candidate; private set => Set(ref _candidate, value); }

    public string Status
    {
        get => _status;
        private set
        {
            if (Set(ref _status, value))
            {
                Raise(nameof(IsOk));
                Raise(nameof(IsFailed));
                Raise(nameof(StatusText));
            }
        }
    }

    public string CandidateStatus
    {
        get => _candidateStatus;
        private set
        {
            if (Set(ref _candidateStatus, value))
            {
                Raise(nameof(HasCandidate));
                Raise(nameof(CandidateFailed));
            }
        }
    }

    public bool IsOk => Status == "ok";
    public bool IsFailed => Status == "failed";
    public bool HasCandidate => CandidateStatus == "ok";
    public bool CandidateFailed => CandidateStatus == "failed";
    public string StatusText => Status switch
    {
        "ok" => "Xong",
        "failed" => "Lỗi",
        _ => "Chưa có",
    };

    public string ErrorText { get => _errorText; private set => Set(ref _errorText, value); }
    public bool WeaponClipped { get => _weaponClipped; private set => Set(ref _weaponClipped, value); }
    public bool IsSelected { get => _isSelected; set => Set(ref _isSelected, value); }

    public void Update(FrameState s)
    {
        Status = s.Status;
        CandidateStatus = s.CandidateStatus;
        ErrorText = s.Error?.Message ?? "";
        WeaponClipped = s.WeaponClipped;
        if (s.Preview != _previewPath)
        {
            _previewPath = s.Preview;
            Image = Load(s.Preview);
        }
        if (s.CandidatePreview != _candidatePath)
        {
            _candidatePath = s.CandidatePreview;
            Candidate = Load(s.CandidatePreview);
        }
    }

    /// <summary>Loads an image fully into memory, so the file is never kept open or cached by path.</summary>
    public static BitmapImage? Load(string? path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
            bmp.UriSource = new Uri(path, UriKind.Absolute);
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
        catch (Exception)
        {
            return null;
        }
    }
}
