import numpy as np
from PIL import Image

from animgen.engines.tagger import MAX_TAGS, THRESHOLD, appearance_tags, prepare


def test_only_confident_appearance_tags_most_confident_first():
    scores = {"1boy": 0.99, "male_focus": 0.97, "solo": 0.96, "white_background": 0.9, "beard": 0.95,
              "standing": 0.9, "holding_sword": 0.8, "arrow_(projectile)": 0.7, "red_cape": 0.6,
              "looking_at_viewer": 0.5, "smile": 0.5, "boots": THRESHOLD - 0.01}
    assert appearance_tags(scores) == ["1boy", "male focus", "beard", "red cape"]


def test_at_most_max_tags():
    assert len(appearance_tags({f"tag_{i}": 0.9 for i in range(40)})) == MAX_TAGS


def test_prepare_pads_the_figure_to_a_white_bgr_square():
    img = Image.new("RGBA", (300, 600), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (100, 100, 200, 500))           # a red figure, 100×400
    x = prepare(img)
    assert x.shape == (1, 448, 448, 3) and x.dtype == np.float32
    assert tuple(x[0, 224, 224]) == (0.0, 0.0, 255.0)          # red as BGR
    assert tuple(x[0, 224, 5]) == (255.0, 255.0, 255.0)        # white padding beside the figure
