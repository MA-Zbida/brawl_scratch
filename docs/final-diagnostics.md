# Frozen-policy video diagnostics

This is a diagnostic, not a win-rate benchmark. Use Wu Shang versus attacking
Easy Mordex on Small Brawlhaven, with continuous respawns and no automatic damage
reset. Keep the same resolution, HUD calibration, and bot settings as training.
Nothing in this tool updates the checkpoint or changes the training reward.

## Record latest, then best

From the repository root in PowerShell:

~~~powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.record_final_diagnostic --model train\models\easy_bot_v0\final_agent\mordex_5h_baseline_20260911-121932_87f330\latest.zip --windows 2 --seconds 120 --video --delay 5
~~~

Then repeat with best.zip instead of latest.zip. Each invocation creates a fresh
directory under outputs/final_diagnostics. The model loads before the countdown.
Both windows are deterministic by default. Windows continue the same live game;
they are not independent physical resets or completed matches.

Send the entire generated directory:

| File | Meaning |
|---|---|
| diagnostic.mp4 | Captured perception images, resized and lossy-encoded, with timing/action header |
| transitions.csv | Full pre/post state and history, all 27 action probabilities, action IDs, timing, reward components and observation provenance |
| events.jsonl | Detection boxes and timestamped controller API calls, joined by interaction |
| windows.csv | Training-room stock/damage/reward summaries |
| manifest.json | Checkpoint SHA-256, checkpoint metadata, settings and alignment contracts |
| summary.json | Completion/interruption status and recorded counts |

Video is opt-in: omit --video for a timing comparison without encoding.
--stochastic samples the frozen distribution rather than taking its argmax.
Do not compare a stochastic recording to a deterministic one as if only the
checkpoint changed. --max-steps is a safety cap per window (default 10000).
Ctrl+C retains completed rows and closes the environment and video writer.
Hardware/encoding errors retain partial diagnostics and propagate the error.

## Synchronization contract

- policy_capture_id identifies the retrieval behind pre_* (the policy input).
- observed_capture_id identifies the retrieval behind post_* (returned state).
- video_frame indexes the MP4, starting at zero. -1 means no encoded image.
- A capture ID counts successful retrievals, not distinct game frames.
- repeated_pixels detects exact image equality, not guaranteed sensor freshness.
- frame_returned_s timestamps the frame-provider return, not exposure or display
  presentation. Capture buffers may already be old.
- first_input_call_s and events.jsonl describe input-controller API calls. They
  do not prove when Brawlhalla consumed a key or whether an attack was legal.
- All *_s timestamps use the same monotonic process clock; wall_s is elapsed time.
- The MP4 contains one frame per valid retrieval at nominal --video-fps (30).
  Its playback timeline is NOT the live timeline. Use the CSV timestamps to
  reconstruct elapsed time. No frames are invented to fill long delays.
- The video uses BGR images from the same capture stream; it does not launch a
  second screen grabber. Header rendering does not modify detector input.
- video_encode_s measures rendering/encoding time, not all diagnostic overhead.
  Image copying, equality checking, probabilities and CSV/JSON writing also cost
  time. Compare with a no-video recording before treating recorded cadence as
  ordinary deployment performance.

The current environment retrieves a frame before injecting the current action.
Consequently post_* must not be assumed to be the result of action_requested.
This recorder preserves that behavior and exposes its ordering. It does not
silently repair the action/observation alignment or relabel world-model replay.
It uses MatchTrainingEnv and the stock_damage_v1 objective from final training,
not the older standalone run_match reward/scoring path.

## First review

1. Visually verify self/opponent identity, weapons and stock events against logs.
2. Inspect decision-frame age, duplicate images and capture/input ordering.
3. Review ignored pickup opportunities: visible, reachable, unarmed, policy
   probability, input issued, possession afterwards.
4. Review a few whiffs and edge deaths with their preceding second of state and
   input history. Do not infer confirmed hits from attack-button presses.
5. Compare latest and best without changing rewards or logits.

Two windows locate failure mechanisms, not statistically establish superiority.
After a targeted fix, use a larger fixed evaluation protocol and report stock
exchanges separately from actual match results.

## Input/HUD correctness gate (2026-09-12)

### Fixed HUD setup contract

The reader deliberately uses fixed slots, without portrait tracking. At 1920x1080,
the current `config.py` calibration expects **Wu Shang on the right** and
**Mordex on the left** in the top-right HUD. This is independent of which side
of the stage either fighter occupies.

Verify that layout before recording or training. The training-room recordings
showed a portrait reorder; warming up is not a guarantee against a later reorder.
If it happens, stop and restore the layout. Reward attribution after a swap is
not trustworthy, even when the RGB damage curve is correct. Do not use those
intervals to rank checkpoints. Keeping fixed slots is an explicit operating
constraint, not an automatic identity check.

### Retained validation fixes

New manifests distinguish `input_contract_version=poll_visible_edges_v2` and
`hud_contract_version=validated_hud_v2`. Final training config, checkpoint metadata,
and world-replay session metadata carry the same versions. Resuming old weights
does not restore the old control/HUD implementation; compare these as new runs.
Reward coefficients remain `stock_damage_v1`.

- Consecutive tap requests cannot extend a down edge indefinitely. Each press
  lasts the configured latch steps and at least 20 ms; its release lasts at least
  20 ms before another press. Directions remain held independently. No sleeps,
  worker threads, or obsolete queued taps are introduced.
- Jump, dodge and pickup inference consumes newly emitted press edges, not every
  repeated request. An emitted edge is still not proof of game acceptance during
  hitstun, cooldown, or an animation. PPO-update cleanup clears the sequencer too.
- Damage keeps the existing RGB curve and fixed coordinates. Unmatched colours
  abstain instead of becoming zero. Changes require three consistent valid samples
  (within two damage units), including true respawn decreases. This adds two capture
  intervals of confirmation latency for a stable change. Stock flashes suppress
  damage reads until 250 ms after the last observed flash.
  The first valid read after an unreadable start establishes the damage baseline
  without generating hit reward.
- The blue/cyan signal at the calibrated player stock pixel means **game-credited
  stock gain**; red means own stock loss regardless of cause. No separate,
  guessed recent-hit timeout is imposed on the game's credit. An uncredited
  opponent suicide is not counted as a credited KO by this one-pixel reader.
  A stable damage reset can still be accepted without blue credit.
- `events.jsonl` now saves `hud` (original sampled RGB, decoded raw damage,
  per-side accepted-sample flags, and raw stock signal) and `input_taps_pressed`.
  This permits exact pixel validation without decoding the lossy MP4. The existing
  CSV `damage_valid` remains the character-existence/respawn guard; per-side HUD
  validity is recorded separately. Rejected samples leave the damage baseline
  unchanged and produce no damage delta.

Before further training, repeat the frozen latest/best commands above, first
deterministic and then with `--stochastic`, preserving the same arena settings.
Inspect both new key edges and actual on-screen response, and verify that KO
flashes create no damage credit. Do not claim a performance improvement from
headless tests alone, or compare video-enabled cadence with ordinary training
without accounting for recording overhead.

## Next gate: weapon possession, then fighting behavior

The target remains one goal-independent policy initialized from skill pretraining,
learning to win against Easy Mordex as Wu Shang on Small Brawlhaven. Training-room
stock exchanges are the training proxy; actual match wins need separate evaluation.
The agent learns tactics from a specified outcome reward, not from an unspecified
objective. Standalone skill scores are not the final success criterion.

The September 12 diagnostic manifests have `weapon_classifier_enabled=false`.
The ground-weapon detector still supplies weapon locations, but possession is
action-inferred. That cannot reliably observe a knockback disarm or an input the
game ignored. This finding is a sensing limitation, not proof that it caused all
of the policy's ignored pickup opportunities.

Use the same explicit setting in frozen diagnostics and final training:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.record_final_diagnostic `
  --model train\models\easy_bot_v0\final_agent\mordex_5h_baseline_20260911-121932_87f330\latest.zip `
  --windows 2 --seconds 60 --device cpu --video `
  --weapon-state-source visual_fusion
```

`--weapon-state-source action` (the default) preserves the existing setup.
`visual_fusion` enables the existing crop classifier every four captures, using
the checkpoint and confidence threshold in `EnvConfig`. Classifier placement is
independent of the CPU policy: its existing automatic device selection prefers
CUDA when available. No new action, observation entry, reward, or forced pickup
is introduced. The setting is saved in run/diagnostic metadata. Video stays opt-in.

Freshness fixes required before this test:

- Cached predictions remain available for diagnostics but cannot overwrite a
  newer pickup/throw inference or repeatedly report a disarm disagreement.
- Missing detections cannot produce fresh crops from old fighter boxes; missing
  fighters/frames invalidate their classifier cache.
- The environment's capture precedes input injection. Visual fusion now happens
  before inference from that newer input. A later fresh image can correct an
  inferred pickup that did not actually execute.

Inspect `player_weapon_visual_valid`, `player_weapon_visual_fresh`, confidence,
and `pre_/post_player_has_weapon` against the video. A visual disagreement is not
by itself proof of a knockback disarm; it can also correct an unexecuted pickup.
Require correct armed/unarmed transitions, including a visible loss and reacquisition,
without persistent false states. If the policy never acquires a weapon, that run
cannot validate possession transitions; do not infer a pass from an uneventful clip.

After this gate, measure opportunity-conditioned pickup selection, repeated-heavy
behavior, and unsafe offstage chasing. Final training currently uses plain PPO;
the transferred initializer is not continuing BC/retention regularization. Check
whether integrated fighting demonstrations or a bounded retention mechanism are
needed before changing reward coefficients, banning heavies, or switching all
training to Hard. Anticipation and follow-up chains are subsequent measured
experiments, not capabilities established by the existing feedforward checkpoint.

## September 13: identity safeguards and exact weapon inputs

`bounded_triangle_v2` retains the blue-triangle identity design, but rejects
near-level character centres, ambiguous triangle matches, and distant or contested
nearest-position fallbacks. Continuation without the triangle expires after 150 ms.
Association compares centres to centres (Memory otherwise stores feet). A detection
reacquired after missing frames starts with zero velocity, rather than reporting the
whole disappearance as one frame of movement. These are provisional conservative
gates with regression coverage, not a claim of perfect tracking.

The classifier now abstains when identity is unobserved. Its cache is pruned even
on cadence-skipped frames. Existing observation dimensions, policy weights, reward,
and fixed HUD ownership are unchanged. Runtime metadata records the identity version.

Offline replay of `latest_20260913-111000_62f446` (2,809 detections):

- Interaction 1255: lone Mordex is no longer adopted as Wu Shang.
- Interaction 1499: Wu Shang beneath the triangle wins the overlap association.
- Interaction 1995: lone Mordex is no longer adopted as off-screen Wu Shang.
- Indicator-assigned coverage: 2,669/2,809 (95.0%), versus 2,707 (96.4%) before.
  Coverage is NOT identity accuracy. Other frames still need live review.
- Local resolver microcheck: approximately 0.009 ms per frame, versus 0.008 ms
  before (median of three full passes). This excludes YOLO, classifier, and video.

The classifier is NOT promoted yet. Native live inputs were not saved in the old
run. Its resized/lossy MP4 produces different predictions when recropped, including
correct unarmed predictions at some previously false-armed frames. That does not
establish why live inference differed and is not justification for retraining on
automatically inferred labels.

Run this short frozen check with opt-in exact inputs:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.record_final_diagnostic `
  --model train\models\easy_bot_v0\final_agent\mordex_5h_baseline_20260911-121932_87f330\latest.zip `
  --windows 2 --seconds 60 --device cpu --video `
  --weapon-state-source visual_fusion --weapon-crops
```

`weapon_crops/unlabelled/*.png` contains the exact native-resolution padded BGR
patches passed to the classifier, before resize. `crops.jsonl` joins them to
`observed_capture_id`, fighter identity, boxes, and predictions; labels remain null.
`contract.json` records preprocessing and classifier device; the run manifest records
the classifier checkpoint SHA256. Crops can include the initial reset capture, which
has no transition row. Normal training and diagnostics without `--weapon-crops`
perform no crop recording. PNG encoding adds diagnostic I/O/latency; do not compare
its frame rate directly against ordinary training.

Include visibly armed and unarmed intervals, disappearance/respawn, and overlap.
If the policy supplies no pickup transition, that part of the gate remains untested.
Next: inspect exact-input errors, test preprocessing/checkpoint parity, then decide
whether targeted manually labelled live crops require classifier fine-tuning. Use
separate encounters/recordings for held-out evaluation; never label from predictions.

### Manual labels with a fixed preview

Open `weapon_label_preview.png` once in the editor, then run:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.label_weapon_crops `
  --crops-dir outputs\final_diagnostics\latest_20260913-115051_f9b536\weapon_crops `
  --out data\weapon_crops_live_20260913 `
  --preview weapon_label_preview.png --scale 4
```

Enter `0` for unarmed, `1` for armed, or `2` for skip/uncertain, then press Enter.
`u` undoes the previous decision in this session; `q` quits. The PNG is replaced
atomically after each decision. Original crops are untouched; labelled copies and
the resumable `labels.jsonl` ledger go to `--out`. Re-run the same command to resume.
Skipped crops are recorded with a null label, not a third training class. The
classifier's prediction is hidden to avoid biasing human labels. Label the fighter
in the crop, not a weapon lying nearby; skip ambiguous overlaps or obscured hands.

YOLO-labelled source images use the same interface with `--dataset brawlhalla-yolo-data`
instead of `--crops-dir`. Use a separate output directory for each recording/source;
the ledger retains recording and capture provenance for later held-out splits.
The classifier trainer's existing frame-based folds do not enforce encounter-level
separation: do not feed mixed live recordings straight into a claimed held-out
benchmark without explicitly separating recordings/encounters first.

### Deploy the reviewed live classifier (2026-09-13)

The reviewed candidate is `train/models/weapon_classifier_r18_live_v1.pt`.
Training and diagnostics accept `--weapon-classifier-checkpoint`; supplying a path
does not enable the classifier by itself: also select `--weapon-state-source visual_fusion`.
Old defaults/checkpoints are preserved. The loader prints the resolved classifier
path and rejects missing files before environment creation. Diagnostic manifests
retain its SHA256; training config and opt-in replay metadata retain its selected path.
The policy remains on CPU; classifier device selection is independent.

First run a short frozen integration check, with no additional crop dataset:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.record_final_diagnostic `
  --model train\models\easy_bot_v0\final_agent\mordex_5h_baseline_20260911-121932_87f330\latest.zip `
  --windows 2 --seconds 60 --device cpu --video `
  --weapon-state-source visual_fusion `
  --weapon-classifier-checkpoint train\models\weapon_classifier_r18_live_v1.pt
```

Use Wu Shang vs attacking Easy Mordex on Small Brawlhaven, in the training room,
with the existing fixed HUD ownership. Confirm weapon flags across armed/unarmed
intervals, respawns and overlaps. Lack of a pickup opportunity is not evidence
that the policy refuses pickups. Recording does not train or modify weights.

After the sensing check passes, continue into a separate, bounded training run:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.final_agent `
  --resume train\models\easy_bot_v0\final_agent\mordex_5h_baseline_20260911-121932_87f330\latest.zip `
  --save-dir train\models\easy_bot_v0\final_agent `
  --model-name weapon_visual_v1_pilot `
  --timesteps 20480 --n-steps 512 --batch-size 256 --device cpu `
  --eval-every 0 `
  --weapon-state-source visual_fusion `
  --weapon-classifier-checkpoint train\models\weapon_classifier_r18_live_v1.pt
```

This is exactly 40 rollout blocks, with unchanged outcome reward and no automatic
evaluation windows. Each invocation creates a unique run directory. It saves CSVs,
checkpoints and training plots, but no video, crops or world replay unless opted in.
Compare its saved `latest.zip` against the frozen baseline using the same diagnostic
command and setup. Inspect pickup opportunities, damage trade, stock exchanges and
unsafe offstage chases, not reward alone. Training-room exchanges are not match wins.
