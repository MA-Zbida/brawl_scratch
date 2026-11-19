"""The recovery reflex (brawl/reflex.py): when it takes over, where it steers, what it presses."""
import numpy as np

from brawl import actions, observation, reflex

F, LEDGE = reflex.F, observation.LEDGE


def state(x_px: float, below: bool, vy: float, jumps: int = 0, heavy_ago=None, dodge_ago=None, last="none",
          airborne=40, hit_ago=None) -> np.ndarray:
    """Wu Shang at ``x_px`` from the stage centre (the ledges at +-350), timers in decisions."""
    o = np.zeros(observation.SIZE, np.float32)
    half = LEDGE / 2
    o[F["agent_present"]] = 1
    o[F["agent_x"]] = x_px / observation.X
    o[F["agent_offstage"]] = float(abs(x_px) > half)
    o[F["agent_below"]] = float(below)
    o[F["agent_vy"]] = vy
    o[F["agent_to_left_ledge"]] = (x_px + half) / LEDGE
    o[F["agent_to_right_ledge"]] = (half - x_px) / LEDGE
    o[F["agent_jumps_since_ground"]] = jumps / 4
    o[F["agent_since_ground"]] = airborne / observation.CLIP
    o[F["agent_since_hit"]] = (hit_ago or observation.CLIP) / observation.CLIP
    o[F["since_press_heavy"]] = (heavy_ago or observation.CLIP) / observation.CLIP
    o[F["since_press_dodge"]] = (dodge_ago or observation.CLIP) / observation.CLIP
    o[F[f"action_{observation.HISTORY}_btn_{last}"]] = 1
    return o


def name(a):
    return None if a is None else actions.name(a)


def test_on_the_stage_or_rising_off_it_the_agent_decides():
    assert reflex.recover(state(0, below=False, vy=0.5)) is None              # over the stage
    assert reflex.recover(state(-500, below=False, vy=-0.5)) is None          # off it, above the top, rising


def test_above_the_top_near_the_ledge_it_drifts_in_and_keeps_its_jumps():
    assert name(reflex.recover(state(-450, below=False, vy=0.3))) == "right+none"
    assert name(reflex.recover(state(650, below=False, vy=0.3))) == "up-left+jump"  # far out: jump anyway


def test_below_it_uses_the_jumps_then_the_recovery_then_the_dodge():
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=1))) == "up-right+jump"
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=1, last="jump"))) == "right+none"  # release first
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=3))) == "up-right+heavy"
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=3, heavy_ago=10))) == "up-right+dodge"
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=3, heavy_ago=10, dodge_ago=5))) == "right+none"
    assert name(reflex.recover(state(-600, below=True, vy=-0.4, jumps=1))) == "right+none"  # still rising: hold


def test_off_the_screen_it_goes_on_from_the_last_place_seen():
    gone = state(-1100, below=True, vy=0.0, jumps=1)  # not detected: last place kept, speed zero
    gone[F["agent_present"]] = 0
    assert name(reflex.recover(gone)) == "up-right+jump"


def test_a_hit_gives_the_recovery_back():
    assert name(reflex.recover(state(-600, below=True, vy=0.3, jumps=3, heavy_ago=30, hit_ago=10))) == "up-right+heavy"


def test_below_the_top_it_climbs_beside_the_ledge_never_under_the_stage():
    assert name(reflex.recover(state(-400, below=True, vy=0.3, jumps=3))) == "up+heavy"        # in the band: straight up
    assert name(reflex.recover(state(-300, below=True, vy=0.3, jumps=3))) == "up-left+heavy"   # under the stage: out
    assert name(reflex.recover(state(320, below=True, vy=0.3, jumps=4, heavy_ago=5, dodge_ago=5))) == "right+none"


def test_it_never_holds_down():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = reflex.recover(state(rng.uniform(-900, 900), bool(rng.integers(2)), rng.uniform(-2, 2), int(rng.integers(5)),
                                 int(rng.integers(1, 80)), int(rng.integers(1, 80)), str(rng.choice(actions.BUTTON_NAMES))))
        assert a is None or "down" not in actions.DIRECTION_NAMES[a[0]]
