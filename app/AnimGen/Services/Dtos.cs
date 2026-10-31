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
    public string? Missing { get; set; }               // null | version | angle: why it cannot be made
    public bool Custom { get; set; }                    // the owner's own pose, made in the pose editor

    /// <summary>Shown after the name when it cannot be made.</summary>
    public string MissingNote => Missing == "angle" ? " (không có ở góc nhìn ba phần tư)" : " (chưa có)";

    public override string ToString() => Ready ? NameVi : NameVi + MissingNote;   // the name screen readers and UI Automation see
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
    public string Angle { get; set; } = "side";             // side | 34 (three-quarter view): the pose set
    public string AngleSource { get; set; } = "detected";   // detected | owner | saved
    public string Name { get; set; } = "";
    public SkeletonInfo? Skeleton { get; set; }
}

/// <summary>The character's skeleton on <see cref="Image"/> (the character's own facing), 18 OpenPose points in
/// image pixels; a missing joint is null.</summary>
// ---- The pose editor (owner, 2026-10-31) ----

/// <summary>The sample character the owner poses: its canvas and its own (bind) joints.</summary>
public sealed class PoseSample
{
    public string Angle { get; set; } = "side";
    public int Width { get; set; }
    public int Height { get; set; }
    public List<double[]?> Rest { get; set; } = new();
    public double Ground { get; set; }
}

/// <summary>One of the 8 frames being posed, in the sample's canvas pixels.</summary>
public sealed class PoseFrame
{
    public List<double[]?> Points { get; set; } = new();
    public string Contact { get; set; } = "ground";        // ground | air
    public List<string> Open { get; set; } = new();         // "r" near hand, "l" far hand
    public List<string> Front { get; set; } = new();        // arm_far, leg_far (3/4 view)
    public List<string> Back { get; set; } = new();         // arm (3/4 view)
    public PoseShape Shape { get; set; } = new();           // squash, part sizes, reach, bends (backend shape.py)
}

/// <summary>A frame's small deformations (owner, 2026-10-31). Neutral: squash 0, sizes and reach 1, bends 0.</summary>
public sealed class PoseShape
{
    public double Squash { get; set; }                         // + shorter and wider, - taller and thinner
    public Dictionary<string, double> Size { get; set; } = new();    // head, hand_r, hand_l, foot_r, foot_l
    public Dictionary<string, double> Reach { get; set; } = new();   // arm_r, arm_l
    public Dictionary<string, double> Bend { get; set; } = new();    // degrees: torso, arm_r, arm_l, leg_r, leg_l
}

public sealed class PoseEditReply
{
    public PoseSample Sample { get; set; } = new();
    public List<PoseFrame> Frames { get; set; } = new();
    public string? Id { get; set; }
    public string? Name { get; set; }
    public string? Grip { get; set; }
}

public sealed class PosePreview
{
    public string Image { get; set; } = "";
    public List<double[]?> Points { get; set; } = new();
}

public sealed class PoseSaved
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
}

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
    public List<string> Editing { get; set; } = new();  // "parts" while the Photoshop file is out for the owner
    public string? Psd { get; set; }                    // the Photoshop file of the stage (null: an older rig)
    public string? Stage { get; set; }                  // "map" (parts as found), "parts" (hidden painted), null: older
}

/// <summary>The Photoshop file of the body parts, for the owner to correct.</summary>
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
