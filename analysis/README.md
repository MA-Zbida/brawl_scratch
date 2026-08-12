# Figure generation

Every figure in the report is generated from run CSVs by a script here. Nothing is
drawn by hand, so the paper can be regenerated from a run directory.

## Usage

```powershell
python -m analysis.plot_learning_curves --metric success
python -m analysis.plot_learning_curves --metric entropy
python -m analysis.plot_retention_matrix
```

Inputs come from `train/models/`, outputs go to `assets/figures/`.

| Script | Reads | Produces |
|---|---|---|
| `plot_learning_curves.py` | `llc_<phase>_episodes.csv` | success / return / goal error / entropy / idle-rate curves |
| `plot_retention_matrix.py` | `llc_<phase>_eval.csv` | retention heatmap, phase trained x phase evaluated |

## Style rules

`style.py` holds the shared style. Three rules are load-bearing and should not be
changed casually:

1. **The categorical order is fixed and validated.** The slot order in
   `style.SERIES` was checked for colour-vision-deficient separation on the
   adjacent-pair list (worst adjacent CVD ΔE 9.1, worst adjacent normal-vision
   ΔE 19.6, on the light surface). Reordering or substituting hues invalidates
   that result — revalidate if you change them.
2. **Colour follows the entity, not the rank.** `style.PHASE_COLOR` pins a colour
   per phase, so plotting a subset of phases never repaints the survivors.
3. **Identity is never colour-alone.** Three slots sit below 3:1 contrast on the
   light surface, so every series carries a direct label in addition to the
   legend.

Other conventions: one y-axis per chart (never a secondary axis), recessive
grid and axes, sequential ramps in a single hue, and the diverging scale
centred on the 0.85 retention gate so passing and failing read as opposite
rather than merely different in intensity.

## Planned figures

Not yet implemented; listed so the report structure is explicit.

- Per-stage step-time breakdown (capture / detect / memory / reward / policy)
- Goal-error distributions per family, before BC / after BC / after PPO
- Ablation bars: −BC, −anchor KL, −replay, −PCGrad, −FiLM
- Detector precision-recall, colour versus grayscale training data
- FiLM latent projection coloured by goal type
- World-model rollout error against horizon
