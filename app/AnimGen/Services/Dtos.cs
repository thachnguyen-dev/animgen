using System.Text.Json.Serialization;

namespace AnimGen.Services;

// Shapes of the backend's JSON replies (snake_case on the wire; see backend/animgen/server.py).

public sealed class ErrorInfo
{
    public string Code { get; set; } = "";
    public string Message { get; set; } = "";
    public Dictionary<string, string>? Params { get; set; }
}

public sealed class ErrorReply
{
    public ErrorInfo? Error { get; set; }
}

public sealed class HealthReply
{
    public bool Ok { get; set; }
    public string Engine { get; set; } = "";
    public bool Demo { get; set; }
    public string Version { get; set; } = "";
}

public sealed class AnimationInfo
{
    public string Id { get; set; } = "";
    public string Animation { get; set; } = "";
    public string? Variant { get; set; }
    public string NameVi { get; set; } = "";
    public bool Ready { get; set; }
    public string Weapon { get; set; } = "never";       // never | optional | required
    public List<string> WeaponKinds { get; set; } = new();
    public string? Status { get; set; }
}

public sealed class Figure
{
    public int Left { get; set; }
    public int Top { get; set; }
    public int Right { get; set; }
    public int Bottom { get; set; }
}

public sealed class CharacterInfo
{
    public string Path { get; set; } = "";
    public int Width { get; set; }
    public int Height { get; set; }
    public int FigureHeight { get; set; }
    public string Facing { get; set; } = "right";
    public string FacingSource { get; set; } = "assumed";   // detected | assumed | owner
    public string Name { get; set; } = "";
}

public sealed class CharacterReply
{
    public CharacterInfo? Character { get; set; }
    public SessionState? State { get; set; }
}

public sealed class FrameState
{
    public int Index { get; set; }
    public string Status { get; set; } = "empty";          // empty | ok | failed
    public int Attempts { get; set; }
    public ErrorInfo? Error { get; set; }
    public bool WeaponClipped { get; set; }
    public string? Preview { get; set; }
    public string CandidateStatus { get; set; } = "none";  // none | ok | failed
    public string? CandidatePreview { get; set; }
}

public sealed class AnimationRef
{
    public string Id { get; set; } = "";
    public string NameVi { get; set; } = "";
    public string? Status { get; set; }
    public string Weapon { get; set; } = "never";
}

public sealed class WeaponRef
{
    public string WeaponId { get; set; } = "";
    public string? Skin { get; set; }
}

public sealed class SessionState
{
    public string Engine { get; set; } = "";
    public bool Demo { get; set; }
    public bool ModelsLoaded { get; set; }
    public CharacterInfo? Character { get; set; }
    public AnimationRef? Animation { get; set; }
    public WeaponRef? Weapon { get; set; }
    public List<FrameState> Frames { get; set; } = new();
    public bool Complete { get; set; }
    public List<int> ExportHeights { get; set; } = new();
}

public sealed class SkinInfo
{
    public string Name { get; set; } = "";
    public List<string> Images { get; set; } = new();
    public List<string> Paths { get; set; } = new();
    public bool Ready { get; set; }
}

public sealed class WeaponInfo
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Kind { get; set; } = "";
    public List<SkinInfo> Skins { get; set; } = new();

    [JsonIgnore]
    public string KindVi => Kind switch
    {
        "one_handed" => "Một tay",
        "two_handed" => "Hai tay",
        "polearm" => "Cán dài",
        "bow" => "Cung",
        "staff" => "Gậy phép",
        _ => Kind,
    };

    public override string ToString() => $"{Name} ({KindVi})";
}

public sealed class TaskInfo
{
    public string Id { get; set; } = "";
    public string Kind { get; set; } = "";
    public string State { get; set; } = "";      // queued | running | completed | failed | cancelled
    public string Stage { get; set; } = "";      // loading_models | generating | retrying | exporting
    public int Frame { get; set; }
    public int FramesTotal { get; set; }
    public int Attempt { get; set; }
    public int Done { get; set; }
    public int Total { get; set; }
    public string Message { get; set; } = "";
    public ErrorInfo? Error { get; set; }
    public List<ErrorInfo> Warnings { get; set; } = new();
    public double Elapsed { get; set; }

    [JsonIgnore]
    public bool Finished => State is "completed" or "failed" or "cancelled";
}

public sealed class TaskReply
{
    public TaskInfo? Task { get; set; }
}

public sealed class PathReply
{
    public string Path { get; set; } = "";
}
