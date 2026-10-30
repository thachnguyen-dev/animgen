using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace AnimGen.Services;

/// <summary>
/// Starts the Python backend (§9.1): a per-run secret token, loopback only, attached to a job object so it dies
/// with the app. The backend prints <c>READY &lt;port&gt;</c> when it is listening.
/// </summary>
public sealed class BackendProcess : IDisposable
{
    private readonly AppPaths _paths;
    private Process? _process;
    private JobObject? _job;
    private StreamWriter? _log;

    public string Token { get; } = Convert.ToHexString(RandomNumberGenerator.GetBytes(24));
    public int Port { get; private set; }

    public BackendProcess(AppPaths paths) => _paths = paths;

    public async Task StartAsync(TimeSpan timeout)
    {
        if (!File.Exists(_paths.Python))
            throw new BackendException("backend_missing", $"Không tìm thấy Python của ứng dụng: {_paths.Python}");
        if (!Directory.Exists(Path.Combine(_paths.Backend, "animgen")))
            throw new BackendException("backend_missing", $"Không tìm thấy phần xử lý: {_paths.Backend}");

        Directory.CreateDirectory(_paths.Logs);
        _log = new StreamWriter(Path.Combine(_paths.Logs, "backend.log"), append: false, new UTF8Encoding(false))
        {
            AutoFlush = true,
        };

        var psi = new ProcessStartInfo
        {
            FileName = _paths.Python,
            WorkingDirectory = _paths.Data,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        psi.ArgumentList.Add("-X");
        psi.ArgumentList.Add("utf8");
        psi.ArgumentList.Add("-m");
        psi.ArgumentList.Add("animgen");
        psi.ArgumentList.Add("--engine");
        psi.ArgumentList.Add(_paths.Engine);
        psi.ArgumentList.Add("--data");
        psi.ArgumentList.Add(_paths.Data);
        psi.ArgumentList.Add("--models");
        psi.ArgumentList.Add(_paths.Models);
        psi.ArgumentList.Add("--parent-pid");
        psi.ArgumentList.Add(Environment.ProcessId.ToString());
        psi.Environment["ANIMGEN_TOKEN"] = Token;
        psi.Environment["PYTHONPATH"] = _paths.Backend;
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        // Offline only (§2.4).
        psi.Environment["HF_HUB_OFFLINE"] = "1";
        psi.Environment["TRANSFORMERS_OFFLINE"] = "1";
        psi.Environment["HF_HUB_DISABLE_TELEMETRY"] = "1";
        psi.Environment["HF_HOME"] = Path.Combine(_paths.Data, "temp", "hf");

        _process = Process.Start(psi) ?? throw new BackendException("backend_start", "Không khởi động được phần xử lý.");
        try
        {
            _job = new JobObject();
            _job.Add(_process);
        }
        catch (Exception ex)
        {
            // The backend also watches the app's process id, so this is not fatal.
            _log.WriteLine($"job object: {ex.Message}");
        }

        _process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            try { _log?.WriteLine(e.Data); }
            catch (ObjectDisposedException) { /* app is closing */ }
        };
        _process.BeginErrorReadLine();

        var readLine = _process.StandardOutput.ReadLineAsync();
        var done = await Task.WhenAny(readLine, Task.Delay(timeout));
        if (done != readLine)
            throw new BackendException("backend_start", "Phần xử lý không phản hồi khi khởi động. Xem logs\\backend.log.");
        string? line = await readLine;
        if (line == null || !line.StartsWith("READY ") || !int.TryParse(line.AsSpan(6), out int port))
            throw new BackendException("backend_start", "Phần xử lý dừng khi khởi động. Xem logs\\backend.log.");
        Port = port;

        // Keep draining stdout so the pipe never fills.
        _ = Task.Run(async () =>
        {
            try
            {
                string? l;
                while ((l = await _process.StandardOutput.ReadLineAsync()) != null)
                    _log?.WriteLine(l);
            }
            catch (Exception) { /* process ended */ }
        });
    }

    public bool HasExited => _process?.HasExited ?? true;

    public void Dispose()
    {
        try
        {
            if (_process is { HasExited: false })
            {
                _process.Kill(entireProcessTree: true);
                _process.WaitForExit(3000);
            }
        }
        catch (Exception) { /* already gone */ }
        _job?.Dispose();
        _process?.Dispose();
        _log?.Dispose();
    }
}
