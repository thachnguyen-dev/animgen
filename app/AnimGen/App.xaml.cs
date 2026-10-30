using System.Globalization;
using System.Windows;
using System.Windows.Markup;
using AnimGen.Services;
using AnimGen.ViewModels;
using AnimGen.Views;

namespace AnimGen;

/// <summary>
/// Starts the window first (so the owner sees something at once), then the backend; a failed start is shown in
/// the window's status line, never as a silent exit. Command line: <c>--demo</c> runs the demo engine (no AI).
/// </summary>
public partial class App : Application
{
    private BackendProcess? _backend;
    private BackendClient? _client;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        var vi = new CultureInfo("vi-VN");
        CultureInfo.DefaultThreadCurrentCulture = vi;
        CultureInfo.DefaultThreadCurrentUICulture = vi;
        FrameworkElement.LanguageProperty.OverrideMetadata(typeof(FrameworkElement),
            new FrameworkPropertyMetadata(XmlLanguage.GetLanguage(vi.IetfLanguageTag)));

        DispatcherUnhandledException += (_, args) =>
        {
            MessageBox.Show(args.Exception.Message, "Lỗi", MessageBoxButton.OK, MessageBoxImage.Error);
            args.Handled = true;
        };

        bool demo = e.Args.Any(a => string.Equals(a, "--demo", StringComparison.OrdinalIgnoreCase));
        var paths = new AppPaths(demo);
        var vm = new MainViewModel();
        var window = new MainWindow(vm);
        MainWindow = window;
        window.Show();

        try
        {
            if (RuntimeRelocation.NeedsRelocation(paths, out _))
            {
                vm.ShowBusy("Thư mục ứng dụng đã được di chuyển. Đang chuẩn bị lại môi trường Python (không cần mạng)…");
                await RuntimeRelocation.RelocateAsync(paths);
            }
            _backend = new BackendProcess(paths);
            await _backend.StartAsync(TimeSpan.FromSeconds(90));
            _client = new BackendClient(_backend.Port, _backend.Token);
            await vm.InitializeAsync(_client);
        }
        catch (BackendException ex)
        {
            vm.ShowStartupError(ex.Message);
        }
        catch (Exception ex)
        {
            vm.ShowStartupError($"Không khởi động được phần xử lý: {ex.Message}");
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        try
        {
            BackendClient? c = _client;
            if (c != null) Task.Run(() => c.ShutdownAsync()).Wait(TimeSpan.FromSeconds(2));
        }
        catch (Exception) { /* closing anyway */ }
        _client?.Dispose();
        _backend?.Dispose();
        base.OnExit(e);
    }
}
