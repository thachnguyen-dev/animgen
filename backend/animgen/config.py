"""Fixed numbers of the pipeline, in one place.

Status of each value follows the design document: anything not Confirmed there is Proposed here and can be tuned
in Phase 0 without touching the rest of the code.
"""

from __future__ import annotations

# Confirmed (§5.3): every animation has exactly 8 frames.
FRAME_COUNT = 8

# Proposed (§10.4): frames are generated at about 1024 px tall, square canvas.
CANVAS_SIZE = 1024
# Proposed: the character fills this fraction of the canvas height, leaving room above for raised weapons
# (a sword held overhead reaches about 0.45 of the figure height above the head).
CHARACTER_HEIGHT_FRACTION = 0.52
# Proposed: where the feet stand (fraction of canvas height from the top) and the anchor x (fraction of width).
FOOT_LINE_FRACTION = 0.92
ANCHOR_X_FRACTION = 0.50

# Export heights offered to the owner. Frames are square, so width = height. Owner (2026-10-30): the sprites are for
# mobile 2D games, so the output can be much smaller; 1024 is gone (it would only be an upscale of the 768 drawing).
EXPORT_HEIGHTS = (128, 256, 512)

# Pending (§8.2), Proposed value 2: automatic retries per frame after a failed technical check.
DEFAULT_AUTO_RETRIES = 2
MAX_AUTO_RETRIES = 5  # hard ceiling: retries are never unbounded

# Input checks (§4.4). Proposed thresholds.
ALPHA_OPAQUE = 128               # alpha above this counts as part of the figure
MIN_TRANSPARENT_FRACTION = 0.05  # at least this share of pixels must be fully transparent
MIN_FIGURE_FRACTION = 0.005      # the figure must cover at least this share of the image
SIGNIFICANT_REGION_FRACTION = 0.05  # opaque regions smaller than this share of the figure are ignored (hair strands)
MIN_INPUT_SIDE = 64
MAX_INPUT_SIDE = 8192

# Frame validation (§8.1). Proposed thresholds for a generated frame.
FRAME_MIN_TRANSPARENT_FRACTION = 0.20
FRAME_MIN_FIGURE_FRACTION = 0.01

# Edge cleaning after background removal (§6.5 step 4).
ALPHA_FLOOR = 8                  # alpha at or below this becomes 0
STRAY_REGION_FRACTION = 0.01     # detached specks smaller than this share of the figure are removed
FIGURE_BOX_MARGIN = 0.15         # detached regions farther than this share of the skeleton's height are removed

# Canvas fit (§6.5 step 5): grounded frames are moved so the feet stand on the foot line, if the drift is small.
MAX_FOOT_SNAP_FRACTION = 0.06    # of the character height

# Hand anchors for weapons attached later in the game. The hand point is the wrist pushed this far along the forearm.
HAND_FROM_WRIST = 0.30           # fraction of the forearm length
