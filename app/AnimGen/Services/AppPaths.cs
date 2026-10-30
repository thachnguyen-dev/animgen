using System.IO;

namespace AnimGen.Services;

/// <summary>
/// Where everything lives. The data folder follows the installed app (§9.1, §14): the app runs from
/// <c>DATA\app</c>, so DATA is the app folder's parent and the whole DATA folder can be moved.
///
/// DATA\app      this program
/// DATA\backend  the Python package "animgen"
/// DATA\python   the bundled Python (uv-managed)
/// DATA\venv     the virtual environment with PyTorch, Diffusers, ...
/// DATA\models   the model folders
/// DATA\weapons  the owner's weapon images and their grip/tip points
/// DATA\temp     this session's frames (removed on the next start)
/// DATA\logs     backend log
///
/// For development, environment variables override the defaults:
/// ANIMGEN_DATA, ANIMGEN_PYTHON, ANIMGEN_BACKEND, ANIMGEN_ENGINE (demo | diffusers).
/// </summary>
public sealed class AppPaths
{
    public string Data { get; }
    public string Backend { get; }
    public string Python { get; }
    public string Models { get; }
    public string Logs { get; }
    public string Engine { get; }

    public AppPaths(bool demo)
    {
        string appDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        string defaultData = Directory.GetParent(appDir)?.FullName ?? appDir;
        Data = Env("ANIMGEN_DATA") ?? defaultData;
        Backend = Env("ANIMGEN_BACKEND") ?? Path.Combine(Data, "backend");
        Python = Env("ANIMGEN_PYTHON") ?? Path.Combine(Data, "venv", "Scripts", "python.exe");
        Models = Path.Combine(Data, "models");
        Logs = Path.Combine(Data, "logs");
        Engine = demo ? "demo" : (Env("ANIMGEN_ENGINE") ?? "diffusers");
    }

    /// <summary>True when this folder looks like a real install (§14: never act as if a dev folder were one).</summary>
    public bool LooksInstalled() =>
        File.Exists(Python) && Directory.Exists(Path.Combine(Backend, "animgen"));

    private static string? Env(string name)
    {
        string? v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(v) ? null : v;
    }
}
