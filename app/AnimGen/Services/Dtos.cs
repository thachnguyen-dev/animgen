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
    public string Grip { get; set; } = "never";         // never | optional | always: hands closed for a weapon
    public List<string> WeaponKinds { get; set; } = new();
    public string? Status { get; set; }

    public override string ToString() => Ready ? NameVi : $"{NameVi} (chưa có)";   // the name screen readers and UI Automation see
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
    public string FacingSource { get; set; } = "assumed";   // detected | assumed | owner | saved
    public string Name { get; set; } = "";
    public SkeletonInfo? Skeleton { get; set; }
}

/// <summary>The character's skeleton on <see cref="Image"/> (the character's own facing), 18 OpenPose points in
/// image pixels; a missing joint is null.</summary>
public sealed class SkeletonInfo
{
    public string? Image { get; set; }
    public int Size { get; set; }
    public List<double[]?> Points { get; set; } = new();
    public string Source { get; set; } = "detected";       // detected | proportions | saved
    public bool Confirmed { get; set; }
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
    public string? Preview { get; set; }
    public string CandidateStatus { get; set; } = "none";  // none | ok | failed
    public string? CandidatePreview { get; set; }
}

public sealed class AnimationRef
{
    public string Id { get; set; } = "";
    public string NameVi { get; set; } = "";
    public string? Status { get; set; }
    public string Grip { get; set; } = "never";
}

/// <summary>A hanging part of the rig (cape, skirt, long hair) and whether it swings.</summary>
public sealed class HangingInfo
{
    public string Name { get; set; } = "";              // cape | skirt | hair
    public bool On { get; set; }
}

/// <summary>The character's rig: the AI drawings every frame is made from (owner, 2026-10-30).</summary>
public sealed class RigInfo
{
    public int Seed { get; set; }
    public string? View { get; set; }                   // the assembled character, swinging parts tinted
    public string? HandView { get; set; }               // the open-hand drawing
    public List<HangingInfo> Hanging { get; set; } = new();
    public List<string> Editing { get; set; } = new();  // drawings with a copy out for the owner's editor (rig | hand)
}

/// <summary>A rig drawing copied out for the owner's own paint program.</summary>
public sealed class EditReply
{
    public string File { get; set; } = "";
    public SessionState? State { get; set; }
}

public sealed class SessionState
{
    public string Engine { get; set; } = "";
    public bool Demo { get; set; }
    public bool ModelsLoaded { get; set; }
    public CharacterInfo? Character { get; set; }
    public AnimationRef? Animation { get; set; }
    public bool Grip { get; set; }
    public bool Finished { get; set; }                  // the frames had the AI finishing pass: they can be redrawn
    public RigInfo? Rig { get; set; }
    public List<FrameState> Frames { get; set; } = new();
    public bool Complete { get; set; }
    public List<int> ExportHeights { get; set; } = new();
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
