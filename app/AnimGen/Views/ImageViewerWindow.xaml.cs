using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace AnimGen.Views;

/// <summary>
/// A picture shown large (owner, 2026-10-31: the step-2 character rig zoomed in): the mouse wheel zooms around the
/// cursor, dragging pans, and fixed steps fit the window or show 100–400 %. Zoomed in, pixels stay sharp.
/// </summary>
public partial class ImageViewerWindow : Window
{
    private const double MinZoom = 0.1;
    private const double MaxZoom = 16;
    private const double WheelStep = 1.2;       // zoom factor per wheel notch
    private const double SharpFrom = 2;         // from this zoom on, pixels are drawn as blocks, not smoothed
    private const double FitMargin = 0.06;      // around the drawn part when fitting, as a share of its size
    private const byte SolidAlpha = 40;         // a pixel this opaque counts as drawn

    private readonly ImageSource _image;
    private readonly Rect _content;             // the drawn part of the picture, in pixels: what "Vừa khung" fits
    private bool _fitted = true;                // keeps fitting when the window is resized, until the owner zooms
    private Point? _grab;

    public ImageViewerWindow(ImageSource image, string title)
    {
        _image = image;
        InitializeComponent();
        Title = $"Xem phóng to — {title}";
        TitleText.Text = title;
        Picture.Source = image;
        _content = DrawnPart(image);
    }

    /// <summary>The box of the pixels that are not transparent, with a margin (the whole picture if none).</summary>
    private static Rect DrawnPart(ImageSource image)
    {
        if (image is not BitmapSource bmp) return new Rect(0, 0, image.Width, image.Height);
        var src = new FormatConvertedBitmap(bmp, PixelFormats.Bgra32, null, 0);
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
        if (right < left) return new Rect(0, 0, w, h);
        double m = Math.Max(right - left, bottom - top) * FitMargin;
        double x0 = Math.Max(0, left - m), y0 = Math.Max(0, top - m);
        return new Rect(x0, y0, Math.Min(w, right + 1 + m) - x0, Math.Min(h, bottom + 1 + m) - y0);
    }

    private void View_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (_fitted) Fit();
    }

    private void Fit()
    {
        if (View.ActualWidth <= 0 || _content.Width <= 0) return;
        double z = Math.Min(View.ActualWidth / _content.Width, View.ActualHeight / _content.Height);
        SetZoom(z, new Point((View.ActualWidth - _content.Width * z) / 2 - _content.X * z,
                             (View.ActualHeight - _content.Height * z) / 2 - _content.Y * z));
        _fitted = true;
    }

    /// <summary>The image is drawn at its pixel size (Stretch None, device-independent units = pixels at 96 dpi)
    /// times the zoom, its top left corner at <paramref name="at"/>.</summary>
    private void SetZoom(double z, Point at)
    {
        z = Math.Clamp(z, MinZoom, MaxZoom);
        // Stretch None draws at the bitmap's own dpi size: scale so 1 zoom = 1 image pixel per screen unit.
        double dpiScale = Picture.Source is BitmapSource b && b.Width > 0 ? b.PixelWidth / b.Width : 1;
        Zoom.ScaleX = Zoom.ScaleY = z * dpiScale;
        Pan.X = at.X;
        Pan.Y = at.Y;
        RenderOptions.SetBitmapScalingMode(Picture, z >= SharpFrom ? BitmapScalingMode.NearestNeighbor : BitmapScalingMode.HighQuality);
        ZoomText.Text = $"{Math.Round(z * 100)}%";
        _fitted = false;
    }

    private double Current => Zoom.ScaleX / (Picture.Source is BitmapSource b && b.Width > 0 ? b.PixelWidth / b.Width : 1);

    /// <summary>Zoom to <paramref name="z"/> keeping the image point under <paramref name="around"/> in place.</summary>
    private void ZoomAround(double z, Point around)
    {
        double old = Current;
        z = Math.Clamp(z, MinZoom, MaxZoom);
        double ix = (around.X - Pan.X) / old, iy = (around.Y - Pan.Y) / old;
        SetZoom(z, new Point(around.X - ix * z, around.Y - iy * z));
    }

    private void Fit_Click(object sender, RoutedEventArgs e) => Fit();

    private void Zoom_Click(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement { Tag: string t } && double.TryParse(t, out double z))
            ZoomAround(z, new Point(View.ActualWidth / 2, View.ActualHeight / 2));
    }

    private void View_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        ZoomAround(e.Delta > 0 ? Current * WheelStep : Current / WheelStep, e.GetPosition(View));
        e.Handled = true;
    }

    private void View_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        _grab = e.GetPosition(View);
        View.CaptureMouse();
    }

    private void View_MouseMove(object sender, MouseEventArgs e)
    {
        if (_grab is not Point from || e.LeftButton != MouseButtonState.Pressed) return;
        Point to = e.GetPosition(View);
        Pan.X += to.X - from.X;
        Pan.Y += to.Y - from.Y;
        _grab = to;
        _fitted = false;
    }

    private void View_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        _grab = null;
        View.ReleaseMouseCapture();
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();
}
