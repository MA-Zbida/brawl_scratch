# Final-agent training against Easy Mordex

Use Wu Shang against **attacking Easy Mordex** on Small Brawlhaven, at the same
resolution and calibrated HUD positions used for skill collection. Disable bot
damage resets or other training cheats that erase damage/stock feedback.
Start in the training arena before running the command.

The default mode is continuous training-room respawns. It measures **stock
exchanges**, not match wins. Full-match success requires an actual-match benchmark
later. One 146-dimensional, goal-free policy chooses among the existing 27 actions.

## First run

Run from the repository root in PowerShell:

~~~powershell
C:\venvs\brawl312\Scripts\python.exe -m train.final_agent --init-from train\models\easy_bot_v0\llc_combat_execution.zip --save-dir train\models\easy_bot_v0\final_agent --model-name mordex_transfer --mode training_room --timesteps 10240 --eval-episodes 3 --device cpu --world-replay world_model\data\final_agent --delay 5
~~~

The run starts with three deterministic evaluation windows, trains stochastically,
then evaluates the final checkpoint. A window is 3,000 observed transitions by
default, so evaluations take real time and are logged separately from training.
--eval-episodes controls the scheduled/end evaluation window count (default 10).
--eval-every 0 disables baseline/scheduled/end evaluations; then no best checkpoint
is selected. --timesteps 0 saves an untrained transferred checkpoint without playing.

Every invocation creates a new timestamped run directory. Its config, initializer,
transfer report, latest checkpoint, periodic checkpoints, evaluation-selected best,
CSV logs and final PNG remain together. Existing skill artifacts are not overwritten.
ppo/progress.csv retains PPO's optimizer diagnostics, including KL, entropy,
policy loss and value loss.

## Continue

Use the exact run directory printed by the first run:

~~~powershell
C:\venvs\brawl312\Scripts\python.exe -m train.final_agent --resume train\models\easy_bot_v0\final_agent\YOUR_RUN_DIRECTORY\latest.zip --save-dir train\models\easy_bot_v0\final_agent --model-name mordex_transfer --timesteps 100000 --device cpu --world-replay world_model\data\final_agent --delay 5
~~~

Use --timesteps 500000 --forever for repeated long blocks. Ctrl+C saves the
current checkpoint and closes replay shards. Checkpoints default to every 10,240
PPO steps; evaluation to every 51,200, at rollout boundaries. Resume preserves
the PPO timestep counter; the new session has its own evaluation-selected best.
Monitor disk availability on long runs: replay is append-only, with no automatic
deletion. Video recording is not enabled.

## Reward and transfer

- +5 per opponent stock lost; -5 per own stock lost.
- +0.01 per damage point dealt, capped at +1 per opponent life.
- -0.005 per damage point received, capped at -0.5 per own life.
- No damage shaping on stock-loss, missing-character or respawn observations.
- No standing, weapon-holding, attack-button, distance, or whiff reward.
- Training-room windows never receive match win/loss bonuses.
- Actual manual matches receive +20/-20 at final stock loss, once.

State/actor weights transfer; goal-specific parameters and the critic are excluded.
Removing goal conditioning changes the network, so the initializer is evaluated.
The old skill BC actions are not replayed automatically into goal-free PPO.

Default PPO: CPU, n_steps=512, batch_size=256, n_epochs=4, learning_rate=1e-4,
gamma=.999, gae_lambda=.95, clip_range=.1, ent_coef=.01, target_kl=.01.
All are CLI-configurable. These are initial experimental settings.

## Accounting and plots

Reporting windows preserve live history and per-life reward budgets. Training-room
stock counters re-arm for unlimited respawns. The observation encodes full stock
availability in this mode; actual-match stock scarcity remains untested.

During optimizer updates keys are released. A neutral refresh transition is
recorded separately before the next PPO action, with measured observation interval.
It is excluded from PPO and marked update_gap in replay. Evaluation transitions
carry an evaluation flag. Replay schema 4 adds reward components and policy version;
world-model consumers should explicitly decide whether to include these intervals.
Replay records state/action/next-state outcomes, not video.

windows.csv contains reward, stocks, damage, usage and timing. evaluations.csv
contains frozen-policy evaluation results. The training-room best checkpoint
maximizes net stocks/minute, using damage trade/second as a tie-breaker. These are
noisy same-opponent selection measurements, not held-out match-win estimates.
Periodic snapshots allow rollback if later training regresses.

~~~powershell
C:\venvs\brawl312\Scripts\python.exe -m analysis.plot_final_training train\models\easy_bot_v0\final_agent\YOUR_RUN_DIRECTORY
~~~

Plots are also generated on a clean/interrupted training exit. Compare stock
differential and damage trade against the initializer before extending the budget.

--mode manual_matches requires a real new 3-stock game initially. After a match
ends it prompts for a manual rematch and gives three seconds to refocus.
It does not automate results screens. Establish calibrated stock-event correctness
before interpreting results as wins.
