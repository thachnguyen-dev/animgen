using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;

namespace AnimGen.Services;

/// <summary>An error the owner should read: a stable code and a Vietnamese message from the backend.</summary>
public sealed class BackendException : Exception
{
    public string Code { get; }

    public BackendException(string code, string message) : base(message) => Code = code;
}

/// <summary>Talks to the backend over loopback HTTP with the per-run token. Every call either returns its
/// result or throws <see cref="BackendException"/> with a message that can be shown as it is.</summary>
public sealed class BackendClient : IDisposable
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    private readonly HttpClient _http;

    public BackendClient(int port, string token)
    {
        _http = new HttpClient(new SocketsHttpHandler { UseProxy = false })
        {
            BaseAddress = new Uri($"http://127.0.0.1:{port}/"),
            Timeout = TimeSpan.FromMinutes(5),
        };
        _http.DefaultRequestHeaders.Add("X-AnimGen-Token", token);
    }

    public Task<HealthReply> HealthAsync() => GetAsync<HealthReply>("health");
    public Task<List<AnimationInfo>> AnimationsAsync() => GetAsync<List<AnimationInfo>>("animations");
    public Task<SessionState> StateAsync() => GetAsync<SessionState>("state");
    public Task<List<WeaponInfo>> WeaponsAsync() => GetAsync<List<WeaponInfo>>("weapons");
    public Task<TaskInfo> TaskAsync(string id) => GetAsync<TaskInfo>($"tasks/{id}");

    public Task<CharacterReply> SetCharacterAsync(string path, string? facing, string? name) =>
        PostAsync<CharacterReply>("character", new { path, facing, name });

    public Task<WeaponInfo> AddWeaponAsync(string name, string kind, string skin, IEnumerable<string> images) =>
        PostAsync<WeaponInfo>("weapons", new { name, kind, skin, images = images.ToArray() });

    public Task<WeaponInfo> SetPointsAsync(string weaponId, string? skin, int stage, double[] grip, double[] tip) =>
        PostAsync<WeaponInfo>("weapons/points", new { weapon_id = weaponId, skin, stage, grip, tip });

    public Task<PathReply> WeaponPreviewAsync(string weaponId, string? skin, int stage) =>
        PostAsync<PathReply>("weapons/preview", new { weapon_id = weaponId, skin, stage });

    public Task<TaskReply> GenerateAsync(string animation, WeaponRef? weapon) =>
        PostAsync<TaskReply>("generate", new { animation, weapon = WeaponBody(weapon) });

    public Task<TaskReply> RegenerateAsync(int? frame) => PostAsync<TaskReply>("regenerate", new { frame });
    public Task<SessionState> AcceptAsync(int frame) => PostAsync<SessionState>("accept", new { frame });
    public Task<SessionState> KeepAsync(int frame) => PostAsync<SessionState>("keep", new { frame });
    public Task<SessionState> AcceptAllAsync() => PostAsync<SessionState>("accept_all", new { });

    public Task<SessionState> SetWeaponAsync(WeaponRef? weapon) =>
        PostAsync<SessionState>("weapon", new { weapon = WeaponBody(weapon) });

    public Task<TaskReply> ExportAsync(string folder, int height) =>
        PostAsync<TaskReply>("export", new { folder, height });

    public Task CancelAsync(string taskId) => PostAsync<JsonElement>($"tasks/{taskId}/cancel", new { });

    public async Task ShutdownAsync()
    {
        try { await PostAsync<JsonElement>("shutdown", new { }); }
        catch (Exception) { /* closing anyway */ }
    }

    private static object? WeaponBody(WeaponRef? w) => w == null ? null : new { weapon_id = w.WeaponId, skin = w.Skin };

    private async Task<T> GetAsync<T>(string path)
    {
        HttpResponseMessage resp;
        try
        {
            resp = await _http.GetAsync(path);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            throw new BackendException("backend_lost", "Mất kết nối với phần xử lý. Hãy khởi động lại ứng dụng.");
        }
        return await ReadAsync<T>(resp);
    }

    private async Task<T> PostAsync<T>(string path, object body)
    {
        HttpResponseMessage resp;
        try
        {
            resp = await _http.PostAsJsonAsync(path, body, Json);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            throw new BackendException("backend_lost", "Mất kết nối với phần xử lý. Hãy khởi động lại ứng dụng.");
        }
        return await ReadAsync<T>(resp);
    }

    private static async Task<T> ReadAsync<T>(HttpResponseMessage resp)
    {
        using (resp)
        {
            string text = await resp.Content.ReadAsStringAsync();
            if (!resp.IsSuccessStatusCode)
            {
                ErrorInfo? err = null;
                try { err = JsonSerializer.Deserialize<ErrorReply>(text, Json)?.Error; }
                catch (JsonException) { }
                throw new BackendException(err?.Code ?? $"http_{(int)resp.StatusCode}",
                                           err?.Message ?? $"Lỗi phần xử lý (HTTP {(int)resp.StatusCode}).");
            }
            try
            {
                return JsonSerializer.Deserialize<T>(text, Json)
                       ?? throw new BackendException("bad_reply", "Phần xử lý trả lời rỗng.");
            }
            catch (JsonException ex)
            {
                throw new BackendException("bad_reply", $"Phần xử lý trả lời không đọc được: {ex.Message}");
            }
        }
    }

    public void Dispose() => _http.Dispose();
}
