using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using AnimGen.Services;
using AnimGen.ViewModels;
using Microsoft.Win32;

namespace AnimGen.Views;

/// <summary>
/// Weapon setup (§4.2): the owner adds the image(s) of a weapon or a new skin, then clicks the grip point and the
/// tip once per image, and checks the weapon in the character's hand. Skins with images of the same size reuse
/// the points already set.
/// </summary>
public partial class WeaponDialog : Window
{
    public sealed class ImageSlot : ObservableObject
    {
        private string? _path;
        public string Label { get; init; } = "";
        public string? Path { get => _path; set { if (Set(ref _path, value)) Raise(nameof(FileName)); } }
        public string FileName => _path == null ? "Chưa chọn" : System.IO.Path.GetFileName(_path);
    }

    private readonly BackendClient _api;
    private readonly ObservableCollection<ImageSlot> _slots = new();
    private WeaponInfo? _weapon;
    private string? _skin;
    private int _stage = 1;
    private Point? _grip;
    private Point? _tip;
    private BitmapSource? _bitmap;

    public string? SavedWeaponId { get; private set; }

    public WeaponDialog(BackendClient api)
    {
        _api = api;
        InitializeComponent();
        ImageSlots.ItemsSource = _slots;
        RebuildSlots();
    }

    private string Kind => (KindBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "one_handed";

    private void KindBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ImageSlots != null) RebuildSlots();
    }

    private void RebuildSlots()
    {
        _slots.Clear();
        if (Kind == "bow")
        {
            _slots.Add(new ImageSlot { Label = "1. Cung nghỉ" });
            _slots.Add(new ImageSlot { Label = "2. Kéo nửa" });
            _slots.Add(new ImageSlot { Label = "3. Kéo căng" });
        }
        else
        {
            _slots.Add(new ImageSlot { Label = "Ảnh vũ khí" });
        }
    }

    private void ChooseImage_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.Tag is not ImageSlot slot) return;
        var dlg = new OpenFileDialog { Title = slot.Label, Filter = "Ảnh PNG hoặc WebP|*.png;*.webp" };
        if (dlg.ShowDialog(this) == true) slot.Path = dlg.FileName;
    }

    private async void SaveImages_Click(object sender, RoutedEventArgs e)
    {
        string name = NameBox.Text.Trim();
        if (name.Length == 0)
        {
            SetStatus(StatusKind.Error, "Hãy nhập tên vũ khí.");
            return;
        }
        if (_slots.Any(s => s.Path == null))
        {
            SetStatus(StatusKind.Error, Kind == "bow" ? "Cung cần đủ 3 ảnh." : "Hãy chọn ảnh vũ khí.");
            return;
        }
        SaveImagesButton.IsEnabled = false;
        try
        {
            _skin = string.IsNullOrWhiteSpace(SkinBox.Text) ? "mặc định" : SkinBox.Text.Trim();
            _weapon = await _api.AddWeaponAsync(name, Kind, _skin, _slots.Select(s => s.Path!));
            SavedWeaponId = _weapon.Id;
            SkinInfo skin = CurrentSkin()!;
            StageBox.Items.Clear();
            string[] stageNames = { "Cung nghỉ", "Kéo nửa", "Kéo căng" };
            for (int i = 1; i <= skin.Paths.Count; i++)
                StageBox.Items.Add(skin.Paths.Count == 3 ? $"Ảnh {i}: {stageNames[i - 1]}" : "Ảnh vũ khí");
            StageBox.SelectedIndex = 0;
            if (skin.Ready)
            {
                SetStatus(StatusKind.Ok, "Đã lưu. Màu này dùng lại điểm cầm và đầu mũi đã đặt trước đó. Có thể bấm “Xem vũ khí trên tay nhân vật”.");
                PreviewButton.IsEnabled = true;
            }
            else
            {
                SetStatus(StatusKind.Info, "Đã lưu ảnh. Giờ hãy đặt điểm cầm và đầu mũi.");
            }
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
        }
        finally
        {
            SaveImagesButton.IsEnabled = true;
        }
    }

    private SkinInfo? CurrentSkin() => _weapon?.Skins.FirstOrDefault(s => s.Name == _skin);

    private void StageBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        SkinInfo? skin = CurrentSkin();
        if (skin == null || StageBox.SelectedIndex < 0) return;
        _stage = StageBox.SelectedIndex + 1;
        _bitmap = FrameItem.Load(skin.Paths[_stage - 1]);
        WeaponImage.Source = _bitmap;
        WeaponImage.Visibility = Visibility.Visible;
        PreviewImage.Visibility = Visibility.Collapsed;
        ClickArea.Visibility = Visibility.Visible;
        AreaTitle.Text = "Bấm để đặt điểm";
        ResetPoints();
    }

    private void ResetPoints_Click(object sender, RoutedEventArgs e) => ResetPoints();

    private void ResetPoints()
    {
        _grip = null;
        _tip = null;
        Markers.Children.Clear();
        ResetPointsButton.IsEnabled = false;
        SavePointsButton.IsEnabled = false;
        PointsHint.Text = "Bấm vào chỗ tay cầm vũ khí (điểm cầm).";
    }

    private void ClickArea_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_bitmap == null || WeaponImage.ActualWidth <= 0 || _tip != null) return;
        Point p = e.GetPosition(WeaponImage);
        if (p.X < 0 || p.Y < 0 || p.X > WeaponImage.ActualWidth || p.Y > WeaponImage.ActualHeight) return;
        // Image pixels from the displayed position (Stretch=Uniform: the element is exactly the drawn image).
        var px = new Point(p.X * _bitmap.PixelWidth / WeaponImage.ActualWidth,
                           p.Y * _bitmap.PixelHeight / WeaponImage.ActualHeight);
        bool isGrip = _grip == null;
        if (isGrip) _grip = px;
        else _tip = px;
        AddMarker(p, isGrip ? "Cầm" : "Mũi", isGrip ? (Brush)FindResource("BrushAccent") : (Brush)FindResource("BrushError"));
        ResetPointsButton.IsEnabled = true;
        if (_tip == null)
        {
            PointsHint.Text = "Giờ bấm vào đầu mũi (đầu xa nhất của vũ khí, với cung là đầu trên).";
        }
        else
        {
            PointsHint.Text = "Đã đặt đủ hai điểm. Bấm “Lưu điểm”.";
            SavePointsButton.IsEnabled = true;
        }
    }

    private void AddMarker(Point at, string text, Brush brush)
    {
        const double r = 7;
        var dot = new Ellipse { Width = r * 2, Height = r * 2, Stroke = brush, StrokeThickness = 3, Fill = Brushes.Transparent };
        Canvas.SetLeft(dot, at.X - r);
        Canvas.SetTop(dot, at.Y - r);
        var label = new TextBlock { Text = text, Foreground = brush, FontWeight = FontWeights.SemiBold };
        Canvas.SetLeft(label, at.X + r + 2);
        Canvas.SetTop(label, at.Y - r - 2);
        Markers.Children.Add(dot);
        Markers.Children.Add(label);
    }

    private async void SavePoints_Click(object sender, RoutedEventArgs e)
    {
        if (_weapon == null || _grip == null || _tip == null) return;
        SavePointsButton.IsEnabled = false;
        try
        {
            _weapon = await _api.SetPointsAsync(_weapon.Id, _skin, _stage,
                                                new[] { _grip.Value.X, _grip.Value.Y },
                                                new[] { _tip.Value.X, _tip.Value.Y });
            SkinInfo skin = CurrentSkin()!;
            if (skin.Ready)
            {
                SetStatus(StatusKind.Ok, "Đã lưu điểm. Hãy xem vũ khí trên tay nhân vật để xác nhận.");
                PreviewButton.IsEnabled = true;
            }
            else
            {
                SetStatus(StatusKind.Info, $"Đã lưu điểm ảnh {_stage}. Hãy chọn ảnh tiếp theo ở danh sách trên và đặt điểm.");
                if (StageBox.SelectedIndex < StageBox.Items.Count - 1) StageBox.SelectedIndex++;
            }
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
            SavePointsButton.IsEnabled = true;
        }
    }

    private async void Preview_Click(object sender, RoutedEventArgs e)
    {
        if (_weapon == null) return;
        try
        {
            PathReply r = await _api.WeaponPreviewAsync(_weapon.Id, _skin, _stage);
            PreviewImage.Source = FrameItem.Load(r.Path);
            PreviewImage.Visibility = Visibility.Visible;
            ClickArea.Visibility = Visibility.Collapsed;
            AreaTitle.Text = "Vũ khí trên tay nhân vật — nếu lệch, bấm “Đặt lại hai điểm” ở mục B";
            SetStatus(StatusKind.Ok, "Nếu vũ khí nằm đúng trong tay, bấm Đóng. Vũ khí sẽ có trong danh sách.");
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Code == "no_character" ? "Hãy chọn ảnh nhân vật ở cửa sổ chính trước để xem thử." : ex.Message);
        }
    }

    private void SetStatus(StatusKind kind, string text)
    {
        StatusBorder.Visibility = Visibility.Visible;
        StatusText.Text = text;
        (string glyph, string brush) = kind switch
        {
            StatusKind.Ok => ("GlyphOk", "BrushSuccess"),
            StatusKind.Error => ("GlyphError", "BrushError"),
            StatusKind.Warning => ("GlyphWarning", "BrushWarning"),
            _ => ("GlyphInfo", "BrushText"),
        };
        StatusGlyph.Text = (string)FindResource(glyph);
        StatusGlyph.Foreground = (Brush)FindResource(brush);
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();
}
