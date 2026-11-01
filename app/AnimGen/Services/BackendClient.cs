using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
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
    public Task<PosePreviewReply> AnimationPreviewAsync(string id) =>
        GetAsync<PosePreviewReply>($"animations/{Uri.EscapeDataString(id)}/preview");
    public Task<SessionState> StateAsync() => GetAsync<SessionState>("state");
    public Task<TaskInfo> TaskAsync(string id) => GetAsync<TaskInfo>($"tasks/{id}");

    public Task<CharacterReply> SetCharacterAsync(string path, string? facing, string? name) =>
        PostAsync<CharacterReply>("character", new { path, facing, name });

    /// <summary>"side" or "34": the pose set the character's animations are made from.</summary>
    public Task<CharacterReply> SetAngleAsync(string angle) =>
        PostAsync<CharacterReply>("character/angle", new { angle });

    public Task<CharacterReply> ConfirmSkeletonAsync(IEnumerable<double[]?> points) =>
        PostAsync<CharacterReply>("character/skeleton", new { points = points.ToArray() });

    /// <summary>The part map cut from the character's own image, and the AI's open hand (about two minutes).</summary>
    /// <param name="part">"parts" or "hand": only that is made again; null: both.</param>
    public Task<TaskReply> MakeRigAsync(int? seed, string? part = null) =>
        PostAsync<TaskReply>("rig", new { seed, part });

    /// <summary>The AI paints what each part of the part map hides (about five minutes).</summary>
    public Task<TaskReply> RebuildRigAsync() => PostAsync<TaskReply>("rig/rebuild", new { });

    /// <summary>Back to the part map; the painted parts are dropped.</summary>
    public Task<SessionState> BackToMapAsync() => PostAsync<SessionState>("rig/map", new { });

    /// <summary>The Photoshop file of the stage (the part map, or the parts) for the owner to correct.</summary>
    public Task<EditReply> EditRigAsync() => PostAsync<EditReply>("rig/edit", new { });

    /// <summary>The rig rebuilt from the edited drawings.</summary>
    public Task<SessionState> ReloadRigAsync() => PostAsync<SessionState>("rig/reload", new { });

    public Task<SessionState> SetHangingAsync(string name, bool on) =>
        PostAsync<SessionState>("rig/hanging", new { name, on });

    /// <param name="grip">Hands closed around a weapon attached later; null where the animation decides.</param>
    /// <param name="finish">The AI finishing pass (slower).</param>
    // The pose editor: a sample character posed in 8 frames, saved as an animation (owner, 2026-10-31).
    public Task<PoseEditReply> StartPoseAsync(string angle, string? from) =>
        PostAsync<PoseEditReply>("poses/start", new { angle, from });
    public Task<PoseEditReply> OpenPoseAsync(string id) => PostAsync<PoseEditReply>("poses/open", new { id });
    public Task<PosePreview> PreviewPoseAsync(string angle, PoseFrame frame) =>
        PostAsync<PosePreview>("poses/preview", new { angle, frame });
    public Task<PoseSaved> SavePoseAsync(string? id, string name, string angle, string grip, IEnumerable<PoseFrame> frames) =>
        PostAsync<PoseSaved>("poses/save", new { id, name, angle, grip, frames = frames.ToArray() });
    public Task DeletePoseAsync(string id) => PostAsync<JsonElement>("poses/delete", new { id });

    public Task<TaskReply> GenerateAsync(string animation, bool? grip, bool finish) =>
        PostAsync<TaskReply>("generate", new { animation, grip, finish });

    public Task<TaskReply> RegenerateAsync(int? frame) => PostAsync<TaskReply>("regenerate", new { frame });
    public Task<SessionState> AcceptAsync(int frame) => PostAsync<SessionState>("accept", new { frame });
    public Task<SessionState> KeepAsync(int frame) => PostAsync<SessionState>("keep", new { frame });
    public Task<SessionState> AcceptAllAsync() => PostAsync<SessionState>("accept_all", new { });

    public Task<TaskReply> ExportAsync(string folder, int height) =>
        PostAsync<TaskReply>("export", new { folder, height });

    public Task CancelAsync(string taskId) => PostAsync<JsonElement>($"tasks/{taskId}/cancel", new { });

    public async Task ShutdownAsync()
    {
        try { await PostAsync<JsonElement>("shutdown", new { }); }
        catch (Exception) { /* closing anyway */ }
    }


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
            // A sized body: PostAsJsonAsync streams it chunked, and the backend reads Content-Length only.
            using var content = new StringContent(JsonSerializer.Serialize(body, Json), Encoding.UTF8, "application/json");
            resp = await _http.PostAsync(path, content);
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
