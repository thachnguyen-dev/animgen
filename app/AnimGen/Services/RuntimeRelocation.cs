using System.Diagnostics;
using System.IO;

namespace AnimGen.Services;

/// <summary>
/// After the whole DATA folder is moved, the virtual environment still points at the old Python path: uv's
/// venv\Scripts\python.exe has the base interpreter's absolute path built in, so editing pyvenv.cfg is not enough
/// (§14, Inherited—Verify). This re-creates the venv launcher offline with the bundled uv, keeping the installed
/// packages.
/// </summary>
public static class RuntimeRelocation
{
    public static bool NeedsRelocation(AppPaths paths, out string? baseHome)
    {
        baseHome = null;
        string cfg = Path.Combine(paths.Data, "venv", "pyvenv.cfg");
        if (!File.Exists(cfg)) return false;
        foreach (string line in File.ReadAllLines(cfg))
        {
            int eq = line.IndexOf('=');
            if (eq < 0) continue;
            if (line[..eq].Trim().Equals("home", StringComparison.OrdinalIgnoreCase))
            {
                baseHome = line[(eq + 1)..].Trim();
                break;
            }
        }
        if (baseHome == null) return false;
        string expectedRoot = Path.GetFullPath(Path.Combine(paths.Data, "python")) + Path.DirectorySeparatorChar;
        return !Path.GetFullPath(baseHome + Path.DirectorySeparatorChar)
                    .StartsWith(expectedRoot, StringComparison.OrdinalIgnoreCase);
    }

    public static async Task RelocateAsync(AppPaths paths)
    {
        string uv = Path.Combine(paths.Data, "uv.exe");
        string pyRoot = Path.Combine(paths.Data, "python");
        string? interpreter = Directory.Exists(pyRoot)
            ? Directory.GetDirectories(pyRoot, "cpython-*")
                       .Select(d => Path.Combine(d, "python.exe"))
                       .FirstOrDefault(File.Exists)
            : null;
        if (!File.Exists(uv) || interpreter == null)
            throw new BackendException("relocate", "Thư mục cài đặt đã bị di chuyển nhưng thiếu uv.exe hoặc Python đi kèm.");

        var psi = new ProcessStartInfo
        {
            FileName = uv,
            WorkingDirectory = paths.Data,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string a in new[] { "venv", Path.Combine(paths.Data, "venv"), "--python", interpreter, "--allow-existing" })
            psi.ArgumentList.Add(a);
        psi.Environment["UV_OFFLINE"] = "1";
        psi.Environment["UV_PYTHON_DOWNLOADS"] = "never";
        psi.Environment["UV_NO_CONFIG"] = "1";

        using Process p = Process.Start(psi) ?? throw new BackendException("relocate", "Không chạy được uv.exe.");
        Task<string> err = p.StandardError.ReadToEndAsync();
        Task<string> outp = p.StandardOutput.ReadToEndAsync();
        await p.WaitForExitAsync();
        if (p.ExitCode != 0)
            throw new BackendException("relocate", $"Không chuẩn bị lại được môi trường Python: {(await err).Trim()} {(await outp).Trim()}");
    }
}
