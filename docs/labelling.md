# Labelling the ground-truth set

Perception is measured against these labels (gate P1), and a new detector is trained on them if
the old one fails. They are made once, so the rules below never change between sessions.

## What the labels are for

| The system needs | From | Labelled here? |
|---|---|---|
| Each fighter's position (body box centre) and identity | Detector | **Yes**: `wu_shang`, `mordex` boxes |
| Free weapons: lying, spawning, flying after a throw | Detector | **Yes**: `weapon` boxes |
| Facing, held weapon, attack and stun states | Fighter crops (the image branch sees them) | No |
| Damage, stocks | HUD reader, checked against the damage numbers (R1) | No |
| Stage edges, blast zones | Fixed geometry, measured once | No |

Facing cannot be derived from velocity (fighters drift and dodge backwards), but it does not need
a label: the 160 px crop around each body shows it, and the network learns it if it matters. The
held weapon is also in the crop (and a held-weapon classifier with its own data exists from the
old work, if it ever has to be explicit).

## Classes

| Class | What |
|---|---|
| `wu_shang` | The agent's legend |
| `mordex` | The bot |
| `weapon` | Every weapon **not held** by a fighter: on the ground, spawning (also while its spawn effect plays), or flying after a throw |

## Box rules

1. **Body only.** Head to feet, arms included, as tight as the visible pixels. **Not** the held
   weapon (spear, scythe, gauntlet swing), trails, hit sparks, dust, dodge afterimages, the KO
   explosion, the indicator triangle.
2. **Behind an effect:** box where the body is, even if an effect covers part of it.
3. **Cut by the screen edge:** box the visible part.
4. **Not visible at all** (KO'd, waiting to respawn): no box.
5. **Visible but unusual** (blinking while respawning, mid-dodge, tumbling after a hit): box it.
6. **Overlapping fighters:** one box each, overlapping is fine.
7. **Weapon:** tight around the weapon's own pixels, not its glow or spawn ring.
8. **Complete:** every fighter and every free weapon in the frame gets a box. Anything without a
   box is taught as "nothing here".
9. **Not gameplay** (a menu or the settings bar open, loading): delete the image.
10. Phantoms and other stage decorations are never labelled.

## The frames

- About **150 frames** in total, from **two sessions**: one against Easy and one against Hard
  (Hard fights closer and overlaps more), 75 drafted from each, spread evenly in time.
- The setting never changes: Brawlhaven (small), Wu Shang vs Mordex, **default skins and colours**,
  **no training overlays**, 1920 x 1080. Frames recorded otherwise do not belong in this set.

## Workflow

1. Record the two sessions (`scripts/record_demos.py`), menu closed before the countdown ends.
2. Draft: `python scripts/prelabel.py <easy recording> <hard recording> --count 75`. The old
   detector proposes the boxes; the output folder is printed.
3. Roboflow: new **object detection** project, classes `wu_shang`, `mordex`, `weapon`; upload the
   whole output folder (images, labels and `data.yaml` together); the drafts appear as annotations.
4. Correct every image by the rules above. Most need a tighter box, some a missing one.
5. Generate a version with **no preprocessing** (remove the default resize) and **no
   augmentation**; export **YOLO** format and unzip it into the same output folder as `export/`.

Roboflow renames images (`<name>_png.rf.<hash>.jpg`) and stores them as JPEG; only its label files
are used. They are matched back to the original lossless stills by name, so every label points to
its recording and row (time, keys held).
