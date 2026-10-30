using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using AnimGen.Services;
using AnimGen.ViewModels;

namespace AnimGen.Views;

/// <summary>
/// The character's skeleton (DECISIONS.md, "Character skeleton set by the owner"): the joints found by the pose
/// reader are shown over the character, the owner drags any that are off and confirms. Confirmed once per image;
/// the backend saves it and uses it for every animation of that image.
/// </summary>
public partial class SkeletonDialog : Window
{
    // OpenPose order, as the backend sends the points. "phải"/"trái" are the character's own sides.
    private static readonly string[] JointNames =
    {
        "Mũi", "Cổ", "Vai phải", "Khuỷu tay phải", "Cổ tay phải", "Vai trái", "Khuỷu tay trái", "Cổ tay trái",
        "Hông phải", "Đầu gối phải", "Cổ chân phải", "Hông trái", "Đầu gối trái", "Cổ chân trái",
        "Mắt phải", "Mắt trái", "Tai phải", "Tai trái",
    };

    private static readonly (int A, int B)[] Bones =
    {
        (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    };

    private static readonly int[] RightSide = { 2, 3, 4, 8, 9, 10, 14, 16 };
    private static readonly int[] LeftSide = { 5, 6, 7, 11, 12, 13, 15, 17 };

    private const double MarkerRadius = 6;
    private const double SelectedRadius = 9;
    private const double MarkerStroke = 3;
    private const double BoneStroke = 2;
    private const double GrabDistance = 14;     // display pixels around a joint that pick it up
    private const double ViewMargin = 0.12;     // around the figure, as a share of its height
    private const byte SolidAlpha = 40;

    private readonly BackendClient _api;
    private readonly Point?[] _original;
    private readonly Point?[] _points;
    private readonly BitmapSource _bitmap;      // the part of the canvas shown: the figure and a margin
    private readonly Vector _offset;            // where that part starts on the canvas (the points' space)
    private readonly int _canvasWidth, _canvasHeight;
    private bool _dragging;

    /// <summary>The backend's reply once the owner confirmed; null when the dialog was closed without it.</summary>
    public CharacterReply? Reply { get; private set; }

    public SkeletonDialog(BackendClient api, SkeletonInfo skeleton)
    {
        _api = api;
        BitmapSource canvas = FrameItem.Load(skeleton.Image)
                  ?? throw new InvalidOperationException("Không đọc được ảnh khung xương. Hãy chọn lại ảnh nhân vật.");
        _canvasWidth = canvas.PixelWidth;
        _canvasHeight = canvas.PixelHeight;
        _original = skeleton.Points.Select(p => p is { Length: 2 } ? new Point(p[0], p[1]) : (Point?)null).ToArray();
        // The figure is small on the square canvas: show only it, so the joints are far enough apart to drag.
        Int32Rect view = FigureView(canvas, _original);
        _bitmap = new CroppedBitmap(canvas, view);
        _offset = new Vector(view.X, view.Y);
        InitializeComponent();
        CharacterImage.Source = _bitmap;
        _points = (Point?[])_original.Clone();
        SourceText.Text = skeleton.Source switch
        {
            "saved" => "Đây là khung xương bạn đã xác nhận trước đó.",
            "proportions" => "Máy không nhận ra tư thế, khung xương được đặt theo tỉ lệ chuẩn — hãy kiểm tra kỹ.",
            _ => "Khung xương do máy nhận ra.",
        };
        RefreshList();
    }

    private int Selected => JointList.SelectedIndex;

    private void RefreshList()
    {
        int keep = JointList.SelectedIndex;
        JointList.Items.Clear();
        for (int i = 0; i < JointNames.Length && i < _points.Length; i++)
            JointList.Items.Add(_points[i] == null ? $"{JointNames[i]} (không có)" : JointNames[i]);
        JointList.SelectedIndex = keep;
        DrawMarkers();
    }

    private void JointList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        int i = Selected;
        RemoveJointButton.IsEnabled = i >= 0 && _points[i] != null;
        JointHint.Text = i < 0
            ? "Chọn một khớp hoặc kéo khớp trên ảnh."
            : _points[i] == null
                ? $"{JointNames[i]}: chưa có. Bấm vào chỗ của khớp này trên ảnh nếu nhìn thấy nó."
                : $"{JointNames[i]}: kéo khớp, hoặc bấm vào chỗ đúng trên ảnh.";
        DrawMarkers();
    }

    private void CharacterImage_SizeChanged(object sender, SizeChangedEventArgs e) => DrawMarkers();

    // Image pixels <-> displayed position (Stretch=Uniform: the element is exactly the drawn image).
    private double Scale => CharacterImage.ActualWidth / _bitmap.PixelWidth;

    private Point ToDisplay(Point px) => new((px.X - _offset.X) * Scale, (px.Y - _offset.Y) * Scale);

    private Point? ToPixels(Point display)
    {
        if (CharacterImage.ActualWidth <= 0) return null;
        double x = Math.Clamp(display.X / Scale, 0, _bitmap.PixelWidth - 1) + _offset.X;
        double y = Math.Clamp(display.Y / Scale, 0, _bitmap.PixelHeight - 1) + _offset.Y;
        return new Point(x, y);
    }

    /// <summary>The box of the visible figure and its joints on the canvas, with a margin.</summary>
    private static Int32Rect FigureView(BitmapSource canvas, IEnumerable<Point?> points)
    {
        var src = new FormatConvertedBitmap(canvas, PixelFormats.Bgra32, null, 0);
        int w = src.PixelWidth, h = src.PixelHeight;
        var px = new byte[w * h * 4];
        src.CopyPixels(px, w * 4, 0);
        int left = w, top = h, right = -1, bottom = -1;
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (px[(y * w + x) * 4 + 3] <= SolidAlpha) continue;
            left = Math.Min(left, x); right = Math.Max(right, x);
            top = Math.Min(top, y); bottom = Math.Max(bottom, y);
        }
        foreach (Point p in points.OfType<Point>())
        {
            left = Math.Min(left, (int)p.X); right = Math.Max(right, (int)Math.Ceiling(p.X));
            top = Math.Min(top, (int)p.Y); bottom = Math.Max(bottom, (int)Math.Ceiling(p.Y));
        }
        if (right < left || bottom < top) return new Int32Rect(0, 0, w, h);
        int m = (int)((bottom - top) * ViewMargin);
        left = Math.Max(0, left - m); top = Math.Max(0, top - m);
        right = Math.Min(w - 1, right + m); bottom = Math.Min(h - 1, bottom + m);
        return new Int32Rect(left, top, right - left + 1, bottom - top + 1);
    }

    private Brush SideBrush(int i) => (Brush)FindResource(
        RightSide.Contains(i) ? "BrushAccent" : LeftSide.Contains(i) ? "BrushSuccess" : "BrushText");

    private void DrawMarkers()
    {
        Markers.Children.Clear();
        if (CharacterImage.ActualWidth <= 0) return;
        var boneBrush = (Brush)FindResource("BrushTextMuted");
        foreach ((int a, int b) in Bones)
        {
            if (a >= _points.Length || b >= _points.Length || _points[a] is not Point pa || _points[b] is not Point pb) continue;
            Point da = ToDisplay(pa), db = ToDisplay(pb);
            Markers.Children.Add(new Line { X1 = da.X, Y1 = da.Y, X2 = db.X, Y2 = db.Y, Stroke = boneBrush, StrokeThickness = BoneStroke });
        }
        for (int i = 0; i < _points.Length; i++)
        {
            if (_points[i] is not Point p) continue;
            Point d = ToDisplay(p);
            bool sel = i == Selected;
            double r = sel ? SelectedRadius : MarkerRadius;
            var dot = new Ellipse
            {
                Width = r * 2, Height = r * 2, Stroke = SideBrush(i), StrokeThickness = MarkerStroke,
                Fill = sel ? (Brush)FindResource("BrushAccentSoft") : Brushes.Transparent,
            };
            Canvas.SetLeft(dot, d.X - r);
            Canvas.SetTop(dot, d.Y - r);
            Markers.Children.Add(dot);
            if (sel)
            {
                var label = new TextBlock
                {
                    Text = JointNames[i], Foreground = SideBrush(i), FontWeight = FontWeights.SemiBold,
                    Background = (Brush)FindResource("BrushSurface"),
                };
                Canvas.SetLeft(label, d.X + r + 2);
                Canvas.SetTop(label, d.Y - r - 2);
                Markers.Children.Add(label);
            }
        }
    }

    private int NearestJoint(Point display)
    {
        int best = -1;
        double bestDist = GrabDistance;
        for (int i = 0; i < _points.Length; i++)
        {
            if (_points[i] is not Point p) continue;
            Point d = ToDisplay(p);
            double dist = (d - display).Length;
            if (dist <= bestDist)
            {
                best = i;
                bestDist = dist;
            }
        }
        return best;
    }

    private void ClickArea_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        Point at = e.GetPosition(CharacterImage);
        int hit = NearestJoint(at);
        if (hit >= 0)
        {
            JointList.SelectedIndex = hit;
            _dragging = ClickArea.CaptureMouse();
            return;
        }
        // Not on a joint: move the selected joint here (this is also how a missing joint is placed).
        if (Selected >= 0) MoveSelected(at);
    }

    private void ClickArea_MouseMove(object sender, MouseEventArgs e)
    {
        if (_dragging && e.LeftButton == MouseButtonState.Pressed) MoveSelected(e.GetPosition(CharacterImage));
    }

    private void ClickArea_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_dragging) ClickArea.ReleaseMouseCapture();
    }

    private void ClickArea_LostMouseCapture(object sender, MouseEventArgs e) => _dragging = false;

    private void MoveSelected(Point display)
    {
        int i = Selected;
        if (i < 0 || ToPixels(display) is not Point px) return;
        bool wasMissing = _points[i] == null;
        _points[i] = px;
        if (wasMissing) RefreshList();
        else DrawMarkers();
        RemoveJointButton.IsEnabled = true;
    }

    private void RemoveJoint_Click(object sender, RoutedEventArgs e)
    {
        if (Selected < 0) return;
        _points[Selected] = null;
        RefreshList();
    }

    private void Reset_Click(object sender, RoutedEventArgs e)
    {
        Array.Copy(_original, _points, _points.Length);
        RefreshList();
        StatusBorder.Visibility = Visibility.Collapsed;
    }

    private async void Confirm_Click(object sender, RoutedEventArgs e)
    {
        ConfirmButton.IsEnabled = false;
        try
        {
            Reply = await _api.ConfirmSkeletonAsync(_points.Select(p => p is Point q ? new[] { q.X, q.Y } : null));
            DialogResult = true;
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
            ConfirmButton.IsEnabled = true;
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
