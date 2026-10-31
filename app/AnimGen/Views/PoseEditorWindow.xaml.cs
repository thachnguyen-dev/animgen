using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using AnimGen.Services;
using AnimGen.ViewModels;

namespace AnimGen.Views;

/// <summary>One cell of the editor's 8-frame strip.</summary>
public sealed class PoseCell : ObservableObject
{
    public int Index { get; init; }
    public string Label => $"Khung {Index + 1}";

    private ImageSource? _image;
    public ImageSource? Image { get => _image; set => Set(ref _image, value); }

    private bool _isSelected;
    public bool IsSelected { get => _isSelected; set => Set(ref _isSelected, value); }
}

/// <summary>Where a new pose starts from: an existing animation, or standing (null id).</summary>
public sealed record PoseBase(string? Id, string Name);

/// <summary>One choice of a deformation value, as the owner reads it.</summary>
public sealed record ShapeStep(double Value, string Text);

/// <summary>
/// The pose editor (owner, 2026-10-31; DECISIONS.md "Pose editor"): the sample character is posed in 8 frames by
/// dragging its joints, each frame drawn by the backend exactly as generation draws it, then saved as an animation
/// for characters seen from the same view. Dragging a joint turns the body part from that joint (its bones keep
/// their lengths), dragging a hip moves the whole body, Shift moves one joint alone.
/// </summary>
public partial class PoseEditorWindow : Window
{
    private const int FrameCount = 8;
    private const int Neck = 1;
    private static readonly int[] Hips = { 8, 11 };
    // Each joint's parent: the joint it turns around. The neck turns around the middle of the hips.
    private static readonly Dictionary<int, int> TurnsAround = new()
    {
        [0] = 1, [14] = 0, [15] = 0, [16] = 14, [17] = 15, [2] = 1, [3] = 2, [4] = 3, [5] = 1, [6] = 5, [7] = 6,
        [9] = 8, [10] = 9, [12] = 11, [13] = 12,
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
    private const double ViewMargin = 0.35;     // around every frame's joints, as a share of the figure's height
    private const double PlayInterval = 125;    // ms per frame: 8 frames per second, as the main window plays

    // The deformations of a frame (backend shape.py): group, key, label. Group "" is the squash of the whole body.
    private static readonly (string Group, string Key, string Label)[] ShapeRows =
    {
        ("", "squash", "Bẹp hoặc giãn cả người"),
        ("size", "head", "Đầu"), ("size", "hand_r", "Bàn tay gần"), ("size", "hand_l", "Bàn tay xa"),
        ("size", "foot_r", "Bàn chân gần"), ("size", "foot_l", "Bàn chân xa"),
        ("reach", "arm_r", "Độ dài tay gần"), ("reach", "arm_l", "Độ dài tay xa"),
        ("bend", "torso", "Uốn thân"), ("bend", "arm_r", "Uốn tay gần"), ("bend", "arm_l", "Uốn tay xa"),
        ("bend", "leg_r", "Uốn chân gần"), ("bend", "leg_l", "Uốn chân xa"),
    };
    private static readonly double[] SquashSteps =
        { -0.2, -0.15, -0.1, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, 0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2 };
    private static readonly double[] SizeSteps = { 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 1, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3, 1.4 };
    private static readonly double[] ReachSteps = { 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.08, 1.1, 1.15, 1.2, 1.25 };
    private static readonly double[] BendSteps = { -25, -20, -15, -10, -8, -6, -5, -4, -2, 0, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25 };
    private static readonly System.Globalization.CultureInfo Vi = new("vi-VN");
    private readonly Dictionary<(string, string), ComboBox> _shapeBoxes = new();
    private bool _showingShape;                  // filling the choices for a frame, not the owner choosing

    private readonly BackendClient _api;
    private readonly IReadOnlyList<PoseBase> _bases;
    private readonly ObservableCollection<PoseCell> _cells = new();
    private readonly DispatcherTimer _timer;
    private string _angle = "side";
    private string? _id;                         // the saved pose being changed; null for a new one
    private List<PoseFrame> _frames = new();
    private List<PoseFrame> _started = new();    // as they were when the editor started, for "Đặt lại khung này"
    private BitmapSource?[] _images = new BitmapSource?[FrameCount];
    private readonly int[] _versions = new int[FrameCount];
    private int _canvasWidth, _canvasHeight;
    private Int32Rect _view;
    private int _current;
    private int _dragJoint = -1;
    private bool _dragFree;
    private bool _dirty;
    private int _playAt;

    /// <summary>The saved pose's id once the owner saved; null when closed without saving.</summary>
    public string? SavedId { get; private set; }

    public PoseEditorWindow(BackendClient api, PoseEditReply start, IReadOnlyList<PoseBase> bases, bool editing)
    {
        _api = api;
        _bases = bases;
        InitializeComponent();
        BuildShapeRows();
        Strip.ItemsSource = _cells;
        for (int i = 0; i < FrameCount; i++) _cells.Add(new PoseCell { Index = i });
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(PlayInterval) };
        _timer.Tick += (_, _) => PlayTick();
        BaseBox.ItemsSource = bases;
        BaseBox.SelectedIndex = 0;
        _id = editing ? start.Id : null;
        if (editing)
        {
            NameBox.Text = start.Name ?? "";
            StartPanel.Visibility = Visibility.Collapsed;
            AngleText.Visibility = Visibility.Visible;
        }
        SelectGrip(start.Grip ?? "never");
        Load(start);
        _dirty = false;
        Loaded += async (_, _) => await PreviewAllAsync();
    }

    // ---------------------------------------------------------------------------------------------------------
    // The frames
    // ---------------------------------------------------------------------------------------------------------
    private void Load(PoseEditReply r)
    {
        _angle = r.Sample.Angle;
        foreach (ComboBoxItem item in AngleBox.Items)
            if ((string)item.Tag == _angle) AngleBox.SelectedItem = item;
        AngleText.Text = _angle == "34" ? "Góc nhìn: ba phần tư." : "Góc nhìn: ngang.";
        LayerPanel.Visibility = _angle == "34" ? Visibility.Visible : Visibility.Collapsed;
        _canvasWidth = r.Sample.Width;
        _canvasHeight = r.Sample.Height;
        _frames = r.Frames.Select(Clone).ToList();
        _started = r.Frames.Select(Clone).ToList();
        _images = new BitmapSource?[FrameCount];
        foreach (PoseCell c in _cells) c.Image = null;
        _view = FigureView(r);
        Select(0);
    }

    private static PoseFrame Clone(PoseFrame f) => new()
    {
        Points = f.Points.Select(p => p == null ? null : new[] { p[0], p[1] }).ToList(),
        Contact = f.Contact, Open = f.Open.ToList(), Front = f.Front.ToList(), Back = f.Back.ToList(),
        Shape = new PoseShape
        {
            Squash = f.Shape.Squash, Size = new(f.Shape.Size), Reach = new(f.Shape.Reach), Bend = new(f.Shape.Bend),
        },
    };

    /// <summary>The part of the canvas shown: every frame's joints with room to move around them.</summary>
    private Int32Rect FigureView(PoseEditReply r)
    {
        var pts = r.Frames.SelectMany(f => f.Points).Concat(r.Sample.Rest).OfType<double[]>().ToList();
        if (pts.Count == 0) return new Int32Rect(0, 0, _canvasWidth, _canvasHeight);
        double top = pts.Min(p => p[1]), bottom = Math.Max(pts.Max(p => p[1]), r.Sample.Ground);
        double left = pts.Min(p => p[0]), right = pts.Max(p => p[0]);
        double m = (bottom - top) * ViewMargin;
        int x0 = (int)Math.Max(0, left - m), y0 = (int)Math.Max(0, top - m);
        int x1 = (int)Math.Min(_canvasWidth - 1, right + m), y1 = (int)Math.Min(_canvasHeight - 1, bottom + m * 0.3);
        return new Int32Rect(x0, y0, Math.Max(1, x1 - x0 + 1), Math.Max(1, y1 - y0 + 1));
    }

    private void Select(int i)
    {
        _current = i;
        for (int k = 0; k < _cells.Count; k++) _cells[k].IsSelected = k == i;
        FrameTitle.Text = $"Khung {i + 1}";
        PoseFrame f = _frames[i];
        GroundBox.IsChecked = f.Contact != "air";
        OpenNearBox.IsChecked = f.Open.Contains("r");
        OpenFarBox.IsChecked = f.Open.Contains("l");
        FarArmFrontBox.IsChecked = f.Front.Contains("arm_far");
        FarLegFrontBox.IsChecked = f.Front.Contains("leg_far");
        NearArmBackBox.IsChecked = f.Back.Contains("arm");
        ShowShape(f.Shape);
        ShowFrame(i);
    }

    private void ShowFrame(int i)
    {
        if (_images[i] != null) FigureImage.Source = _images[i];
        DrawMarkers();
    }

    private async Task PreviewAllAsync()
    {
        for (int i = 0; i < FrameCount; i++) await PreviewAsync(i);
    }

    /// <summary>The backend draws the frame as generation will, and says where the joints ended up (bones at
    /// their lengths, grounded feet on the ground).</summary>
    private async Task PreviewAsync(int i)
    {
        int version = ++_versions[i];
        try
        {
            PosePreview p = await _api.PreviewPoseAsync(_angle, _frames[i]);
            if (version != _versions[i]) return;          // moved again meanwhile: a newer drawing is coming
            _frames[i].Points = p.Points;
            BitmapSource? full = FrameItem.Load(p.Image);
            if (full == null) return;
            _images[i] = new CroppedBitmap(full, _view);
            _images[i]!.Freeze();
            _cells[i].Image = _images[i];
            if (i == _current && !_timer.IsEnabled) ShowFrame(i);
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
        }
    }

    private void Cell_Click(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: PoseCell c })
        {
            StopPlay();
            Select(c.Index);
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Drawing the joints
    // ---------------------------------------------------------------------------------------------------------
    private void FigureImage_SizeChanged(object sender, SizeChangedEventArgs e) => DrawMarkers();

    private double Scale => FigureImage.ActualWidth / _view.Width;

    private Point ToDisplay(double[] px) => new((px[0] - _view.X) * Scale, (px[1] - _view.Y) * Scale);

    private Point? ToPixels(Point display)
    {
        if (FigureImage.ActualWidth <= 0) return null;
        double x = Math.Clamp(display.X / Scale, 0, _view.Width - 1) + _view.X;
        double y = Math.Clamp(display.Y / Scale, 0, _view.Height - 1) + _view.Y;
        return new Point(x, y);
    }

    private Brush SideBrush(int i) => (Brush)FindResource(
        RightSide.Contains(i) ? "BrushAccent" : LeftSide.Contains(i) ? "BrushSuccess" : "BrushText");

    private void DrawMarkers()
    {
        Markers.Children.Clear();
        if (FigureImage.ActualWidth <= 0 || _frames.Count == 0 || _timer.IsEnabled) return;
        var muted = (Brush)FindResource("BrushTextMuted");
        if (_current > 0)                                 // the frame before, faint, to pose the motion against
            DrawBones(_frames[_current - 1].Points, muted, 0.45, new DoubleCollection { 3, 3 });
        DrawBones(_frames[_current].Points, muted, 1.0, null);
        List<double[]?> pts = _frames[_current].Points;
        for (int i = 0; i < pts.Count; i++)
        {
            if (pts[i] is not double[] p) continue;
            Point d = ToDisplay(p);
            bool sel = i == _dragJoint;
            double r = sel ? SelectedRadius : MarkerRadius;
            var dot = new Ellipse
            {
                Width = r * 2, Height = r * 2, Stroke = SideBrush(i), StrokeThickness = MarkerStroke,
                Fill = sel ? (Brush)FindResource("BrushAccentSoft") : Brushes.Transparent,
            };
            Canvas.SetLeft(dot, d.X - r);
            Canvas.SetTop(dot, d.Y - r);
            Markers.Children.Add(dot);
        }
    }

    private void DrawBones(List<double[]?> pts, Brush brush, double opacity, DoubleCollection? dash)
    {
        foreach ((int a, int b) in Bones)
        {
            if (a >= pts.Count || b >= pts.Count || pts[a] is not double[] pa || pts[b] is not double[] pb) continue;
            Point da = ToDisplay(pa), db = ToDisplay(pb);
            Markers.Children.Add(new Line
            {
                X1 = da.X, Y1 = da.Y, X2 = db.X, Y2 = db.Y, Stroke = brush, StrokeThickness = BoneStroke,
                Opacity = opacity, StrokeDashArray = dash,
            });
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Dragging
    // ---------------------------------------------------------------------------------------------------------
    private int NearestJoint(Point display)
    {
        int best = -1;
        double bestDist = GrabDistance;
        List<double[]?> pts = _frames[_current].Points;
        for (int i = 0; i < pts.Count; i++)
        {
            if (pts[i] is not double[] p) continue;
            double dist = (ToDisplay(p) - display).Length;
            if (dist <= bestDist)
            {
                best = i;
                bestDist = dist;
            }
        }
        return best;
    }

    private void DragArea_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_frames.Count == 0) return;
        StopPlay();
        int hit = NearestJoint(e.GetPosition(FigureImage));
        if (hit < 0) return;
        _dragJoint = hit;
        _dragFree = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);
        DragArea.CaptureMouse();
        DrawMarkers();
    }

    private void DragArea_MouseMove(object sender, MouseEventArgs e)
    {
        if (_dragJoint < 0 || e.LeftButton != MouseButtonState.Pressed) return;
        if (ToPixels(e.GetPosition(FigureImage)) is not Point to) return;
        MoveJoint(_frames[_current].Points, _dragJoint, to, _dragFree);
        _dirty = true;
        DrawMarkers();
    }

    private async void DragArea_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_dragJoint < 0) return;
        DragArea.ReleaseMouseCapture();
        await PreviewAsync(_current);
    }

    private void DragArea_LostMouseCapture(object sender, MouseEventArgs e)
    {
        _dragJoint = -1;
        DrawMarkers();
    }

    /// <summary>Shift: the joint alone. A hip: the whole body. Any other joint: the body part from it turns
    /// around its parent joint, the joints beyond it with it.</summary>
    private static void MoveJoint(List<double[]?> pts, int joint, Point to, bool free)
    {
        if (pts[joint] is not double[] at) return;
        if (free)
        {
            pts[joint] = new[] { to.X, to.Y };
            return;
        }
        if (Hips.Contains(joint))
        {
            double dx = to.X - at[0], dy = to.Y - at[1];
            for (int i = 0; i < pts.Count; i++)
                if (pts[i] is double[] p) pts[i] = new[] { p[0] + dx, p[1] + dy };
            return;
        }
        double[]? pivot = joint == Neck ? Pelvis(pts) : TurnsAround.TryGetValue(joint, out int par) ? pts[par] : null;
        if (pivot == null) return;
        double turn = Math.Atan2(to.Y - pivot[1], to.X - pivot[0]) - Math.Atan2(at[1] - pivot[1], at[0] - pivot[0]);
        double cos = Math.Cos(turn), sin = Math.Sin(turn);
        foreach (int i in Beyond(joint))
        {
            if (pts[i] is not double[] p) continue;
            double x = p[0] - pivot[0], y = p[1] - pivot[1];
            pts[i] = new[] { pivot[0] + x * cos - y * sin, pivot[1] + x * sin + y * cos };
        }
    }

    private static double[]? Pelvis(List<double[]?> pts) =>
        pts[8] is double[] a && pts[11] is double[] b ? new[] { (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 } : null;

    /// <summary>The joint and every joint that hangs from it.</summary>
    private static IEnumerable<int> Beyond(int joint)
    {
        yield return joint;
        foreach ((int child, int parent) in TurnsAround)
            if (parent == joint && !(joint == Neck && Hips.Contains(child)))
                foreach (int k in Beyond(child)) yield return k;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Frame options
    // ---------------------------------------------------------------------------------------------------------
    private async void FrameOption_Click(object sender, RoutedEventArgs e)
    {
        PoseFrame f = _frames[_current];
        f.Contact = GroundBox.IsChecked == true ? "ground" : "air";
        f.Open = new List<string>();
        if (OpenNearBox.IsChecked == true) f.Open.Add("r");
        if (OpenFarBox.IsChecked == true) f.Open.Add("l");
        f.Front = new List<string>();
        if (FarArmFrontBox.IsChecked == true) f.Front.Add("arm_far");
        if (FarLegFrontBox.IsChecked == true) f.Front.Add("leg_far");
        f.Back = NearArmBackBox.IsChecked == true ? new List<string> { "arm" } : new List<string>();
        _dirty = true;
        await PreviewAsync(_current);
    }

    // ---------------------------------------------------------------------------------------------------------
    // Deformations of the frame
    // ---------------------------------------------------------------------------------------------------------
    private void BuildShapeRows()
    {
        for (int r = 0; r < ShapeRows.Length; r++)
        {
            (string group, string key, string label) = ShapeRows[r];
            ShapeGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var text = new TextBlock { Text = label, FontWeight = FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center };
            text.Margin = (Thickness)FindResource("SpaceGap");
            var box = new ComboBox { DisplayMemberPath = nameof(ShapeStep.Text), Tag = (group, key) };
            AutomationProperties.SetName(box, label);
            box.SelectionChanged += ShapeBox_SelectionChanged;
            Grid.SetRow(text, r);
            Grid.SetRow(box, r);
            Grid.SetColumn(box, 2);
            ShapeGrid.Children.Add(text);
            ShapeGrid.Children.Add(box);
            _shapeBoxes[(group, key)] = box;
        }
    }

    /// <summary>A bend pushes the middle of the part forward (+) or back (-): for the torso that is an arched back
    /// or a hunch.</summary>
    private static string StepText(string group, string key, double v) => group switch
    {
        "" => Math.Abs(v) < 1e-9 ? "Không" : $"{(v > 0 ? "Bẹp" : "Giãn")} {Math.Abs(v * 100).ToString("0.#", Vi)}%",
        "bend" when Math.Abs(v) < 1e-9 => "Không",
        "bend" when key == "torso" => $"{(v > 0 ? "Ưỡn" : "Khom")} {Math.Abs(v).ToString("0.#", Vi)}°",
        "bend" => $"{(v > 0 ? "Ra trước" : "Ra sau")} {Math.Abs(v).ToString("0.#", Vi)}°",
        _ => $"{(v * 100).ToString("0.#", Vi)}%",
    };

    /// <summary>The choices of each row for this frame, its own value among them even when it is not a usual
    /// step (a built-in pose may use one).</summary>
    private void ShowShape(PoseShape s)
    {
        _showingShape = true;
        ShapeTitle.Text = $"Biến dạng khung {_current + 1}";
        foreach (((string group, string key), ComboBox box) in _shapeBoxes)
        {
            double value = ShapeValue(s, group, key);
            double[] steps = group switch { "" => SquashSteps, "size" => SizeSteps, "reach" => ReachSteps, _ => BendSteps };
            var items = steps.Append(value).Select(v => Math.Round(v, 4)).Distinct().OrderBy(v => v)
                             .Select(v => new ShapeStep(v, StepText(group, key, v))).ToList();
            box.ItemsSource = items;
            box.SelectedItem = items.First(it => Math.Abs(it.Value - Math.Round(value, 4)) < 1e-9);
        }
        _showingShape = false;
    }

    private static double ShapeValue(PoseShape s, string group, string key) => group switch
    {
        "" => s.Squash,
        "size" => s.Size.GetValueOrDefault(key, 1.0),
        "reach" => s.Reach.GetValueOrDefault(key, 1.0),
        _ => s.Bend.GetValueOrDefault(key, 0.0),
    };

    private async void ShapeBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_showingShape || sender is not ComboBox { Tag: (string group, string key), SelectedItem: ShapeStep step }) return;
        PoseShape s = _frames[_current].Shape;
        // A neutral value is left out, so a saved pose carries only what was changed.
        Dictionary<string, double>? values = group switch { "size" => s.Size, "reach" => s.Reach, "bend" => s.Bend, _ => null };
        double neutral = group is "size" or "reach" ? 1.0 : 0.0;
        if (values == null) s.Squash = step.Value;
        else if (Math.Abs(step.Value - neutral) < 1e-9) values.Remove(key);
        else values[key] = step.Value;
        _dirty = true;
        await PreviewAsync(_current);
    }

    private async void CopyPrevious_Click(object sender, RoutedEventArgs e)
    {
        int from = (_current + FrameCount - 1) % FrameCount;     // frame 1 takes frame 8: the motion loops
        _frames[_current] = Clone(_frames[from]);
        _dirty = true;
        Select(_current);
        await PreviewAsync(_current);
    }

    private async void ResetFrame_Click(object sender, RoutedEventArgs e)
    {
        _frames[_current] = Clone(_started[_current]);
        Select(_current);
        await PreviewAsync(_current);
    }

    private async void Restart_Click(object sender, RoutedEventArgs e)
    {
        if (_dirty && !Ask("Bắt đầu lại sẽ bỏ dáng đã chỉnh ở cả 8 khung. Tiếp tục?")) return;
        string angle = (string)((ComboBoxItem)AngleBox.SelectedItem).Tag;
        var from = (PoseBase?)BaseBox.SelectedItem;
        StopPlay();
        SetStatus(StatusKind.Busy, "Đang chuẩn bị nhân vật mẫu…");
        try
        {
            Load(await _api.StartPoseAsync(angle, from?.Id));
            _dirty = false;
            StatusBorder.Visibility = Visibility.Collapsed;
            await PreviewAllAsync();
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
        }
    }

    private void SelectGrip(string grip)
    {
        foreach (ComboBoxItem item in GripBox.Items)
            if ((string)item.Tag == grip) GripBox.SelectedItem = item;
    }

    private void GripBox_SelectionChanged(object sender, SelectionChangedEventArgs e) => _dirty = true;

    private void NameBox_TextChanged(object sender, TextChangedEventArgs e) => _dirty = true;

    // ---------------------------------------------------------------------------------------------------------
    // Playing the 8 frames
    // ---------------------------------------------------------------------------------------------------------
    private void Play_Click(object sender, RoutedEventArgs e)
    {
        if (_timer.IsEnabled)
        {
            StopPlay();
            return;
        }
        _playAt = _current;
        _timer.Start();
        PlayGlyph.Text = (string)FindResource("GlyphPause");
        PlayText.Text = "Dừng";
        Markers.Children.Clear();
    }

    private void PlayTick()
    {
        _playAt = (_playAt + 1) % FrameCount;
        if (_images[_playAt] != null) FigureImage.Source = _images[_playAt];
    }

    private void StopPlay()
    {
        if (!_timer.IsEnabled) return;
        _timer.Stop();
        PlayGlyph.Text = (string)FindResource("GlyphPlay");
        PlayText.Text = "Phát";
        ShowFrame(_current);
    }

    // ---------------------------------------------------------------------------------------------------------
    // Saving
    // ---------------------------------------------------------------------------------------------------------
    private async void Save_Click(object sender, RoutedEventArgs e)
    {
        StopPlay();
        SaveButton.IsEnabled = false;
        try
        {
            string grip = (string)((ComboBoxItem)GripBox.SelectedItem).Tag;
            PoseSaved s = await _api.SavePoseAsync(_id, NameBox.Text, _angle, grip, _frames);
            _id = s.Id;
            SavedId = s.Id;
            _dirty = false;
            SetStatus(StatusKind.Ok, $"Đã lưu tư thế “{s.Name}”. Nó có trong danh sách hoạt ảnh.");
        }
        catch (BackendException ex)
        {
            SetStatus(StatusKind.Error, ex.Message);
        }
        finally
        {
            SaveButton.IsEnabled = true;
        }
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        StopPlay();
        if (_dirty && !Ask("Tư thế có thay đổi chưa lưu. Đóng và bỏ các thay đổi?")) e.Cancel = true;
    }

    private bool Ask(string question) =>
        MessageBox.Show(this, question, Title, MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes;

    private void SetStatus(StatusKind kind, string text)
    {
        StatusBorder.Visibility = Visibility.Visible;
        StatusText.Text = text;
        (string glyph, string brush) = kind switch
        {
            StatusKind.Ok => ("GlyphOk", "BrushSuccess"),
            StatusKind.Error => ("GlyphError", "BrushError"),
            StatusKind.Warning => ("GlyphWarning", "BrushWarning"),
            StatusKind.Busy => ("GlyphBusy", "BrushAccent"),
            _ => ("GlyphInfo", "BrushText"),
        };
        StatusGlyph.Text = (string)FindResource(glyph);
        StatusGlyph.Foreground = (Brush)FindResource(brush);
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();
}
