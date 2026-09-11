"""Plot a final-agent campaign: python -m analysis.plot_final_training RUN_DIR."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.io import read_csv


def plot_run(directory: Path) -> Path | None:
    directory = Path(directory)
    if not (directory / "windows.csv").exists():
        return None
    data = read_csv(directory / "windows.csv")
    train = data["usage"] == "train"
    interactions = np.cumsum(data["interactions"])[train]
    minutes = np.maximum(data["seconds"][train] / 60., 1e-6)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, column, title in (
        (axes[0, 0], "stock_differential", "Training net stocks / minute"),
        (axes[0, 1], "damage_trade", "Training damage trade / minute"),
    ):
        values = data[column][train] / minutes
        ax.plot(interactions, values, alpha=.3, linewidth=1, label="Reporting window")
        if len(values):
            width = min(10, len(values))
            ax.plot(interactions[width-1:],
                    np.convolve(values, np.ones(width)/width, mode="valid"),
                    label=f"{width}-window mean")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(title=title, xlabel="Recorded interactions (includes evaluations/gaps)")
        ax.legend(fontsize=8)
    path = directory / "evaluations.csv"
    if path.exists():
        evaluations = read_csv(path)
        for ax, key, title in (
            (axes[1, 0], "net_stocks_per_minute", "Deterministic evaluation: net stocks / minute"),
            (axes[1, 1], "damage_trade", "Deterministic evaluation: damage trade / minute"),
        ):
            values = evaluations[key]
            if key == "damage_trade":
                values = 60 * values / np.maximum(evaluations["seconds"], 1e-6)
            ax.plot(evaluations["training_step"], values, "o-")
            ax.set(title=title, xlabel="PPO training transitions")
            ax.axhline(0, color="black", linewidth=.7)
    else:
        for ax in axes[1]:
            ax.text(.5, .5, "Evaluation not run", ha="center", transform=ax.transAxes)
    mode = str(data["mode"][0])
    fig.suptitle("Final agent — " + (
        "training-room exchanges (not match win rate)" if mode == "training_room"
        else "manual-match stock and damage outcomes"))
    output = directory / "training_progress.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    output = plot_run(args.run_directory)
    if output is None:
        parser.error("No completed reporting windows in this run")
    print(output)


if __name__ == "__main__":
    main()
