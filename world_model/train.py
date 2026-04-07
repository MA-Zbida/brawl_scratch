"""Train the world model and produce diagnostic plots.

Usage:
    python -m world_model.train --data world_model/data/transitions.npz
    python -m world_model.train --data world_model/data/transitions.npz --epochs 100 --rollout-len 20
    python -m world_model.train --data world_model/data/transitions.npz --min-delta-std 1e-3

Outputs (in --out-dir):
    world_model.pt           – trained model weights
    loss_curve.png           – train/val loss over epochs
    per_feature_error.png    – bar chart of MSE per feature
    temporal_rollout.png     – model-rollout vs real trajectory for N random episodes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from feature_extractor.memory.state_spec import StateSpec
from world_model.model import WorldModel, encode_actions, STATE_DIM


# ── data helpers ────────────────────────────────────────────────────────────

def load_data(path: str) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def build_datasets(
    data: dict[str, np.ndarray],
    val_frac: float = 0.15,
    min_delta_std: float = 1e-3,
):
    """Return train/val datasets with robust delta normalisation stats.

    Returns:
        train_ds, val_ds, d_mean, d_std_raw, d_std_used
    """
    states = data["states"]         # (N, 51)
    actions = data["actions"]       # (N, 4)
    next_states = data["next_states"]  # (N, 51)
    dones = data["dones"]           # (N,)

    # Exclude terminal transitions (next_state after done is a reset obs)
    mask = ~dones
    states, actions, next_states = states[mask], actions[mask], next_states[mask]

    deltas = next_states - states   # (N, 51)
    action_oh = encode_actions(actions)  # (N, 12)

    # Normalise deltas with a std floor so near-constant features do not
    # dominate the loss with huge inverse-variance weights.
    d_mean = deltas.mean(axis=0).astype(np.float32)
    d_std_raw = deltas.std(axis=0).astype(np.float32)
    d_std = np.maximum(d_std_raw, float(min_delta_std)).astype(np.float32)

    deltas_norm = (deltas - d_mean) / d_std

    n = len(states)
    idx = np.random.permutation(n)
    split = int(n * (1 - val_frac))

    def _make(i):
        return TensorDataset(
            torch.as_tensor(states[i], dtype=torch.float32),
            torch.as_tensor(action_oh[i], dtype=torch.float32),
            torch.as_tensor(deltas_norm[i], dtype=torch.float32),
        )

    train_ds = _make(idx[:split])
    val_ds = _make(idx[split:])
    return train_ds, val_ds, d_mean, d_std_raw, d_std


# ── training loop ───────────────────────────────────────────────────────────

def train_model(
    model: WorldModel,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    d_mean: np.ndarray,
    d_std: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
) -> dict:
    """Train and return history for both normalised and raw-delta losses."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # loss is on normalised deltas — model.net outputs raw, so we normalise in the loop
    dm = torch.as_tensor(d_mean, dtype=torch.float32, device=device)
    ds = torch.as_tensor(d_std, dtype=torch.float32, device=device)

    history: dict[str, list[float]] = {
        "train_loss_norm": [],
        "val_loss_norm": [],
        "train_mse_raw": [],
        "val_mse_raw": [],
    }

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_norm, total_raw, count = 0.0, 0.0, 0
        for s, a, d_target in train_loader:
            s, a, d_target = s.to(device), a.to(device), d_target.to(device)
            d_pred_raw = model.predict_delta(s, a)
            d_pred_norm = (d_pred_raw - dm) / ds
            loss_norm = nn.functional.mse_loss(d_pred_norm, d_target)
            d_target_raw = d_target * ds + dm
            loss_raw = nn.functional.mse_loss(d_pred_raw, d_target_raw)
            opt.zero_grad()
            loss_norm.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_norm += loss_norm.item() * s.size(0)
            total_raw += loss_raw.item() * s.size(0)
            count += s.size(0)
        scheduler.step()
        train_loss_norm = total_norm / max(count, 1)
        train_mse_raw = total_raw / max(count, 1)

        # ── val ──
        model.eval()
        total_norm, total_raw, count = 0.0, 0.0, 0
        with torch.no_grad():
            for s, a, d_target in val_loader:
                s, a, d_target = s.to(device), a.to(device), d_target.to(device)
                d_pred_raw = model.predict_delta(s, a)
                d_pred_norm = (d_pred_raw - dm) / ds
                loss_norm = nn.functional.mse_loss(d_pred_norm, d_target)
                d_target_raw = d_target * ds + dm
                loss_raw = nn.functional.mse_loss(d_pred_raw, d_target_raw)
                total_norm += loss_norm.item() * s.size(0)
                total_raw += loss_raw.item() * s.size(0)
                count += s.size(0)
        val_loss_norm = total_norm / max(count, 1)
        val_mse_raw = total_raw / max(count, 1)

        history["train_loss_norm"].append(train_loss_norm)
        history["val_loss_norm"].append(val_loss_norm)
        history["train_mse_raw"].append(train_mse_raw)
        history["val_mse_raw"].append(val_mse_raw)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"norm(train/val)={train_loss_norm:.2e}/{val_loss_norm:.2e}  "
                f"raw(train/val)={train_mse_raw:.2e}/{val_mse_raw:.2e}"
            )

    return history


# ── per-feature error ───────────────────────────────────────────────────────

def compute_per_feature_mse(
    model: WorldModel,
    data: dict[str, np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Return (51,) array of per-feature MSE on non-terminal transitions."""
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    dones = data["dones"]

    mask = ~dones
    states, actions, next_states = states[mask], actions[mask], next_states[mask]

    model.to(device)
    model.eval()
    pred = model.predict_np(states, actions)
    mse = np.mean((pred - next_states) ** 2, axis=0)  # (51,)
    return mse


def plot_per_feature_error(mse: np.ndarray, out_path: Path):
    names = StateSpec.FEATURES
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, mse, color="steelblue", edgecolor="black", linewidth=0.3)

    # Highlight top-5 worst features
    top5 = np.argsort(mse)[-5:]
    for i in top5:
        bars[i].set_color("tomato")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("MSE (raw state units)")
    ax.set_title("World Model — Per-Feature Prediction Error")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved per-feature error plot → {out_path}")


# ── temporal rollout ────────────────────────────────────────────────────────

def find_episode_starts(dones: np.ndarray) -> list[int]:
    """Return indices where episodes start (after a done or at index 0)."""
    starts = [0]
    done_idx = np.where(dones)[0]
    for di in done_idx:
        if di + 1 < len(dones):
            starts.append(di + 1)
    return starts


def temporal_rollout_test(
    model: WorldModel,
    data: dict[str, np.ndarray],
    rollout_len: int = 10,
    n_rollouts: int = 8,
    device: str = "cpu",
) -> dict:
    """Model-rollout vs real trajectory comparison.

    Returns dict with:
        real_trajs  : list of (rollout_len+1, 51) arrays
        pred_trajs  : list of (rollout_len+1, 51) arrays
        per_step_mse: (rollout_len,) average MSE at each rollout step
        per_step_feature_mse: (rollout_len, 51) feature-level MSE at each step
    """
    states = data["states"]
    actions = data["actions"]
    dones = data["dones"]

    episode_starts = find_episode_starts(dones)
    # Find episodes long enough for rollout
    valid_starts = []
    for s in episode_starts:
        # Check that there are rollout_len consecutive non-terminal steps
        end = s + rollout_len
        if end >= len(states):
            continue
        if np.any(dones[s:end]):
            continue
        valid_starts.append(s)

    if len(valid_starts) == 0:
        print(f"WARNING: No episodes long enough for {rollout_len}-step rollout")
        return {}

    rng = np.random.default_rng(42)
    chosen = rng.choice(valid_starts, size=min(n_rollouts, len(valid_starts)), replace=False)

    real_trajs, pred_trajs = [], []
    all_errors = np.zeros((rollout_len, STATE_DIM), dtype=np.float64)

    model.to(device)
    model.eval()
    for start in chosen:
        real_traj = states[start: start + rollout_len + 1].copy()  # (L+1, 51)
        pred_traj = np.zeros_like(real_traj)
        pred_traj[0] = real_traj[0]  # start from real s_0

        s = real_traj[0:1]  # (1, 51)
        for t in range(rollout_len):
            a = actions[start + t: start + t + 1]  # (1, 4)
            s = model.predict_np(s, a)  # (1, 51)
            pred_traj[t + 1] = s[0]

        real_trajs.append(real_traj)
        pred_trajs.append(pred_traj)
        all_errors += (pred_traj[1:] - real_traj[1:]) ** 2

    n = len(chosen)
    per_step_feature_mse = all_errors / n          # (rollout_len, 51)
    per_step_mse = per_step_feature_mse.mean(axis=1)  # (rollout_len,)

    return {
        "real_trajs": real_trajs,
        "pred_trajs": pred_trajs,
        "per_step_mse": per_step_mse,
        "per_step_feature_mse": per_step_feature_mse,
    }


def _safe_feature_index(name: str, fallback: str) -> tuple[int, str]:
    names = set(StateSpec.names())
    if name in names:
        return StateSpec.index(name), name
    if fallback not in names:
        raise KeyError(f"Neither '{name}' nor fallback '{fallback}' exist in StateSpec")
    print(f"WARNING: feature '{name}' not found; using '{fallback}' instead")
    return StateSpec.index(fallback), fallback


def plot_temporal_rollout(rollout: dict, out_path: Path, rollout_len: int = 10):
    if not rollout:
        return

    real_trajs = rollout["real_trajs"]
    pred_trajs = rollout["pred_trajs"]
    per_step_mse = rollout["per_step_mse"]
    per_step_feature_mse = rollout["per_step_feature_mse"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (0,0) MSE vs rollout step ──
    ax = axes[0, 0]
    steps = np.arange(1, rollout_len + 1)
    ax.plot(steps, per_step_mse, "o-", color="crimson", linewidth=2)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean MSE")
    ax.set_title("Prediction Error vs Rollout Horizon")
    ax.grid(alpha=0.3)

    # ── (0,1) Heatmap: per-feature error across rollout steps ──
    ax = axes[0, 1]
    # Show top-10 worst features at final step
    final_mse = per_step_feature_mse[-1]
    top10 = np.argsort(final_mse)[-10:][::-1]
    heatmap_data = per_step_feature_mse[:, top10].T  # (10, rollout_len)
    im = ax.imshow(heatmap_data, aspect="auto", cmap="hot")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([StateSpec.FEATURES[i] for i in top10], fontsize=8)
    ax.set_xlabel("Rollout step")
    ax.set_title("Top-10 Noisiest Features (MSE heatmap)")
    fig.colorbar(im, ax=ax)

    # ── (1,0) Example rollout: player_x ──
    ax = axes[1, 0]
    feat_idx = StateSpec.index("player_x")
    for i, (real, pred) in enumerate(zip(real_trajs[:3], pred_trajs[:3])):
        t = np.arange(len(real))
        ax.plot(t, real[:, feat_idx], "o-", alpha=0.7, label=f"real {i}")
        ax.plot(t, pred[:, feat_idx], "x--", alpha=0.7, label=f"pred {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("player_x")
    ax.set_title("Temporal Rollout — player_x")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ── (1,1) Example rollout: opponent damage feature ──
    ax = axes[1, 1]
    feat_idx2, feat_name2 = _safe_feature_index("opponent_damage_pct", fallback="player_damage_pct")
    for i, (real, pred) in enumerate(zip(real_trajs[:3], pred_trajs[:3])):
        t = np.arange(len(real))
        ax.plot(t, real[:, feat_idx2], "o-", alpha=0.7, label=f"real {i}")
        ax.plot(t, pred[:, feat_idx2], "x--", alpha=0.7, label=f"pred {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel(feat_name2)
    ax.set_title(f"Temporal Rollout — {feat_name2}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(f"World Model Temporal Rollout Analysis ({rollout_len}-step horizon)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved temporal rollout plot → {out_path}")


def plot_loss_curve(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(history["train_loss_norm"], label="train", linewidth=1.5)
    ax.plot(history["val_loss_norm"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised)")
    ax.set_title("Training Loss (Normalised Delta)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["train_mse_raw"], label="train", linewidth=1.5)
    ax.plot(history["val_mse_raw"], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (raw delta units)")
    ax.set_title("Training Loss (Raw Delta)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved loss curve → {out_path}")


# ── summary report ──────────────────────────────────────────────────────────

def print_summary(mse: np.ndarray, rollout: dict):
    names = StateSpec.FEATURES
    print("\n" + "=" * 60)
    print("WORLD MODEL DIAGNOSIS REPORT")
    print("=" * 60)

    print("\n── Per-Feature MSE (top-10 worst) ──")
    order = np.argsort(mse)[::-1]
    for rank, i in enumerate(order[:10], 1):
        print(f"  {rank:2d}. {names[i]:30s}  MSE = {mse[i]:.2e}")

    print(f"\n── Overall ──")
    print(f"  Mean MSE across features : {mse.mean():.2e}")
    print(f"  Median MSE               : {np.median(mse):.2e}")
    print(f"  Max MSE feature           : {names[order[0]]} ({mse[order[0]]:.2e})")

    if rollout:
        pms = rollout["per_step_mse"]
        print(f"\n── Temporal Rollout ──")
        print(f"  1-step MSE  : {pms[0]:.2e}")
        if len(pms) >= 5:
            print(f"  5-step MSE  : {pms[4]:.2e}  (drift ratio: {pms[4]/max(pms[0],1e-9):.1f}x)")
        print(f"  {len(pms)}-step MSE : {pms[-1]:.2e}  (drift ratio: {pms[-1]/max(pms[0],1e-9):.1f}x)")

        if pms[-1] / max(pms[0], 1e-9) > 10:
            print("  ⚠  HIGH DRIFT — error explodes over rollout (likely noisy/chaotic features)")
        elif pms[-1] / max(pms[0], 1e-9) > 3:
            print("  ⚡ MODERATE DRIFT — some features accumulate error")
        else:
            print("  ✓  LOW DRIFT — model is temporally stable")

    print("=" * 60)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train world model and produce diagnostics")
    parser.add_argument("--data", type=str, required=True, help="Path to transitions.npz")
    parser.add_argument("--out-dir", type=str, default="world_model/results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--min-delta-std",
        type=float,
        default=1e-3,
        help="Std floor used when normalizing per-feature deltas",
    )
    parser.add_argument("--rollout-len", type=int, default=10)
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    print(f"Loading data from {args.data}")
    data = load_data(args.data)
    n = len(data["states"])
    print(f"  {n:,} transitions, {data['dones'].sum()} terminal")

    # ── build datasets ──
    train_ds, val_ds, d_mean, d_std_raw, d_std = build_datasets(
        data,
        min_delta_std=args.min_delta_std,
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    clipped = int(np.sum(d_std_raw < args.min_delta_std))
    print(
        f"  Delta std floor: {args.min_delta_std:.2e} "
        f"(clipped {clipped}/{len(d_std_raw)} features)"
    )

    # Save normalization stats (needed for MBRL integration)
    np.savez(
        str(out_dir / "delta_stats.npz"),
        mean=d_mean,
        std_raw=d_std_raw,
        std_used=d_std,
        min_std=np.array(args.min_delta_std, dtype=np.float32),
    )

    # ── train ──
    model = WorldModel(hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model, train_ds, val_ds, d_mean, d_std,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
    )

    # Save model
    torch.save(model.state_dict(), str(out_dir / "world_model.pt"))
    print(f"Saved model → {out_dir / 'world_model.pt'}")

    # ── plots ──
    plot_loss_curve(history, out_dir / "loss_curve.png")

    mse = compute_per_feature_mse(model, data, device=args.device)
    plot_per_feature_error(mse, out_dir / "per_feature_error.png")

    rollout = temporal_rollout_test(
        model, data, rollout_len=args.rollout_len, n_rollouts=args.n_rollouts, device=args.device,
    )
    plot_temporal_rollout(rollout, out_dir / "temporal_rollout.png", rollout_len=args.rollout_len)

    # ── summary ──
    print_summary(mse, rollout)


if __name__ == "__main__":
    main()
