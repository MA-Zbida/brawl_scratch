"""The labelling tool for the fighter-state network: clips are cut where Mordex is seen, and drawn."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import label_states  # noqa: E402

from brawl import observation  # noqa: E402


def fake_run(root: Path, shards=2, n=100):
    (root / "trace").mkdir(parents=True)
    f = observation.FEATURES
    for s in range(shards):
        obs = np.zeros((n, observation.SIZE), np.float16)
        obs[:, f.index("opponent_present")] = 1
        obs[40:50, f.index("opponent_present")] = 0  # not seen: no clip may include these
        events = np.zeros((n, 6), np.float32)
        events[70, 1] = 5  # Wu Shang hit
        crops = np.zeros((n, 2, 96, 96), np.uint8)
        crops[:, 1] = np.arange(n, dtype=np.uint8)[:, None, None]  # Mordex's crop carries its index
        np.savez_compressed(root / "trace" / f"shard_{s * n + 1:07d}.npz", step=np.arange(s * n + 1, s * n + n + 1), obs=obs,
                            events=events, crops=crops, features=np.array(f))


def test_clips_are_cut_around_seen_decisions_and_cover_the_moments_before_a_hit(tmp_path):
    run = tmp_path / "runs" / "night" / "default" / "seed0"
    fake_run(run)
    path = label_states.prepare([run], 60, tmp_path / "out", log=lambda m: None)
    z = np.load(path)
    assert z["crops"].shape[1:] == (label_states.LENGTH, 96, 96)
    centre = z["crops"][:, label_states.BEFORE, 0, 0].astype(int)  # the labelled decision's index in its shard
    first = z["crops"][:, 0, 0, 0].astype(int)
    assert ((centre - first) == label_states.BEFORE).all()  # consecutive frames, in order
    assert not np.any((first <= 49) & (first + label_states.LENGTH - 1 >= 40))  # never over the unseen stretch
    assert set(z["stratum"]) >= {"before_hit", "anywhere"} and (z["run"] == "night").all()
    assert ((z["step"] - 1) % 100 == centre).all()


def test_labels_resume_and_undo(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("clip,label,run,step,time\n3,attacking,a,1,0\n4,neutral,a,2,0\n4,undo,a,2,0\n5,hurt,a,3,0\n", encoding="utf-8")
    assert label_states.read_labels(p) == {3: "attacking", 5: "hurt"}


def test_the_window_draws():
    clip = np.random.default_rng(0).integers(0, 255, (label_states.LENGTH, 96, 96), dtype=np.uint8)
    im = label_states.render(clip, 7, "clip 1/10")
    assert im.ndim == 3 and im.shape[2] == 3 and im.shape[1] == 150 * label_states.LENGTH
