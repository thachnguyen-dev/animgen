import numpy as np

from animgen.cloth import Hanging, bend, find_hanging, ground_speed, simulate
from animgen.skeleton import LANK, RANK, Skeleton

FIG = 600.0


def cape(length=300.0) -> Hanging:
    return Hanging("cape", "torso", np.array([480.0, 400.0]), length, np.ones(1))


def test_still_body_leaves_the_cape_hanging():
    angles = simulate([np.array([480.0, 400.0])] * 8, [0.0] * 8, cape(), FIG, 0.0, loop=True)
    assert max(abs(a) for a in angles) < 1


def test_running_makes_the_cape_trail_and_faster_trails_more():
    pivots = [np.array([480.0, 400.0])] * 8
    walk = simulate(pivots, [0.0] * 8, cape(), FIG, 1.3, loop=True)
    run = simulate(pivots, [0.0] * 8, cape(), FIG, 2.2, loop=True)
    assert 5 < np.mean(walk) < np.mean(run) < 45


def test_lean_forward_is_taken_back_so_the_cape_still_hangs_down():
    angles = simulate([np.array([480.0, 400.0])] * 8, [12.0] * 8, cape(), FIG, 0.0, loop=True)
    assert all(-12.5 < a < -9 for a in angles)          # turned back against the body's 12 degree lean


def test_stopping_swings_the_cape_forward_a_little():
    xs = [400, 440, 480, 520, 540, 545, 545, 545]      # the body moves forward and stops
    angles = simulate([np.array([x, 400.0]) for x in xs], [0.0] * 8, cape(), FIG, 0.0, loop=False)
    assert max(angles[:4]) > 2 and min(angles[4:]) < max(angles[:4])


def test_a_cape_near_the_floor_lies_on_it_along_the_body():
    # The body lies on its back (the torso turned -90 degrees), the cape's pivot 50 px above the floor.
    angles = simulate([np.array([480.0, 750.0])] * 8, [-90.0] * 8, cape(), FIG, 0.0, loop=False, ground=800.0)
    for a in angles:
        world = np.radians(a - 90.0)                     # the swing is relative to the turned body
        assert 300.0 * np.cos(world) <= 50.0 + 1e-6      # the hem is not below the floor


def test_bend_keeps_the_pivot_and_turns_the_hem():
    part = Hanging("cape", "torso", np.array([0.0, 0.0]), 100.0, np.array([0.0, 1.0]))
    out = bend(np.array([[0.0, 0.0], [0.0, 100.0]]), part, 90)
    assert np.allclose(out[0], [0, 0]) and np.allclose(out[1], [-100, 0], atol=1e-6)   # hem toward the back (-x)


def test_ground_speed_from_the_planted_foot():
    frames = []
    for i in range(8):
        pts = [None] * 18
        pts[RANK] = (500 - 10 * i, 900)                 # planted, sliding back 10 px a frame
        pts[LANK] = (500 + 10 * i, 850)                 # lifted
        frames.append(Skeleton(pts))
    assert abs(ground_speed(frames[:4], FIG) - 10 * 12 / FIG) < 1e-6
    still = [Skeleton([None] * 10 + [(500, 900)] + [None] * 7)] * 4
    assert ground_speed(still, FIG) == 0


def test_cape_behind_the_back_is_found_hair_needs_length():
    h = w = 1024
    torso = np.zeros((h, w), bool)
    torso[360:580, 480:520] = True                      # the body
    torso[400:700, 400:470] = True                      # a cape well behind it (the figure faces right)
    head = np.zeros((h, w), bool)
    head[260:360, 480:550] = True
    bind = {"torso": ((500.0, 360.0), (500.0, 580.0)), "head": ((500.0, 360.0), (520.0, 300.0))}
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    verts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    found = {p.name: p for p in find_hanging(verts, torso, head, bind, FIG)}
    assert "cape" in found and "hair" not in found
    hem = np.argmin(np.hypot(verts[:, 0] - 432, verts[:, 1] - 696))
    top = np.argmin(np.hypot(verts[:, 0] - 432, verts[:, 1] - 408))
    core = np.argmin(np.hypot(verts[:, 0] - 504, verts[:, 1] - 456))
    assert found["cape"].factor[hem] > 0.8 and found["cape"].factor[top] < 0.2 and found["cape"].factor[core] < 0.1
