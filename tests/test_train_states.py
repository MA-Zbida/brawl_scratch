"""The fighter-state network: free targets from a trace, packing the labels, and a tiny run end to end."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_states  # noqa: E402

from brawl import observation  # noqa: E402

F = observation.FEATURES


def shard(path: Path, n=60, start=1):
    obs = np.zeros((n, observation.SIZE), np.float16)
    obs[:, F.index("since_press_light")] = 3 / observation.CLIP  # light pressed 3 decisions ago: bin 1
    obs[:, F.index("agent_since_hit")] = 1.0                      # not hit for 80: the last bin
    obs[:, F.index("opponent_since_hit")] = 1 / observation.CLIP  # Mordex just hit: bin 0
    events = np.zeros((n, 6), np.float32)
    events[20, 1] = 4.0  # Wu Shang hit at decision 20
    crops = np.random.default_rng(start).integers(0, 255, (n, 2, 96, 96), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, step=np.arange(start, start + n), obs=obs, events=events, crops=crops, features=np.array(F))
    return obs, events


def test_free_targets_read_the_timers_and_the_next_half_second(tmp_path):
    obs, events = shard(tmp_path / "s.npz")
    agent, opp = train_states.free_targets(obs, events, list(F))
    assert (agent[:, 1] == 1).all() and (agent[:, 5] == len(train_states.BINS)).all()
    soon = opp[:, 0]
    assert (soon[11:21] == 1).all() and (soon[:11] == 0).all() and (soon[21:51] == 0).all()  # hit at 20: seen from 11 on
    assert (soon[51:] == -1).all()  # the shard ends before the next half second does
    assert (opp[:, 1] == 0).all()


def test_a_tiny_run_writes_the_report_and_the_network(tmp_path):
    for night in ("n4", "n5"):
        for k in range(2):
            shard(tmp_path / "pre" / night / f"shard_{k:07d}.npz", start=1 + 60 * k)
    rng = np.random.default_rng(0)
    m = 80
    np.savez(tmp_path / "labels.npz", crops=rng.integers(0, 255, (m, train_states.T, 96, 96), dtype=np.uint8),
             label=np.arange(m) % len(train_states.STATES), run=np.array(["n4"] * m),
             step=np.arange(m) * train_states.SEGMENT // 8)  # spread over the folds
    assert train_states.main(["--labels", str(tmp_path / "labels.npz"), "--pretrain", str(tmp_path / "pre"),
                              "--pretrain-steps", "3", "--steps", "3", "--out", str(tmp_path / "out")]) == 0
    text = (tmp_path / "out" / "results" / "states" / "fighter_state.md").read_text(encoding="utf-8")
    assert "context, pretrained" in text and "100 ms ahead" in text and "AUC" in text
    assert (tmp_path / "out" / "states" / "state_net.pt").exists() and (tmp_path / "out" / "states" / "pretrained.pt").exists()


def test_the_labels_pack_keeps_the_frame_and_the_five_before_it(tmp_path, monkeypatch):
    import label_states
    import pack_states

    n = 6
    crops = np.zeros((n, label_states.LENGTH, 96, 96), np.uint8)
    crops[:] = np.arange(label_states.LENGTH, dtype=np.uint8)[None, :, None, None]  # each frame carries its position
    np.savez(tmp_path / "clips.npz", crops=crops, run=np.array(["a"] * n), step=np.arange(n), stratum=np.array(["x"] * n),
             before=label_states.BEFORE, after=label_states.AFTER)
    (tmp_path / "labels.csv").write_text("clip,label,run,step,time\n0,attacking,a,0,0\n2,unclear,a,2,0\n4,hurt,a,4,0\n",
                                         encoding="utf-8")
    d = pack_states.labelled(tmp_path / "clips.npz", tmp_path / "labels.csv")
    assert d["clip"].tolist() == [0, 4] and d["label"].tolist() == [1, 4]
    assert d["crops"].shape == (2, train_states.T, 96, 96)
    assert d["crops"][0, -1, 0, 0] == label_states.BEFORE and d["crops"][0, 0, 0, 0] == label_states.BEFORE - train_states.T + 1


def test_the_pretraining_folder_links_every_fourth_shard_without_copying(tmp_path):
    import os

    import pack_states

    runs = []
    for night in ("advised_night4", "advised_night5"):
        run = tmp_path / "runs" / night / "default" / "seed0"
        for k in range(9):
            shard(run / "trace" / f"shard_{k * 60 + 1:07d}.npz", start=k * 60 + 1)
        runs.append(run)
    out = pack_states.link_shards(runs, tmp_path / "states_pretrain")
    assert [p.name for p in out][:3] == ["advised_night4_shard_0000001.npz", "advised_night4_shard_0000241.npz",
                                          "advised_night4_shard_0000481.npz"] and len(out) == 6
    assert os.stat(out[0]).st_nlink == 2  # the same file, not a copy
