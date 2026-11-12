# Labelling the detector's data

The detector is the agent's only source of where things are and what each fighter holds: the
structured vector, the fighter crops, weapon ownership and every gate downstream are computed from
its boxes and points. It is trained and measured (gate P1) on frames labelled by the rules below.
The rules are fixed once (revised 2026-11-10: weapon-inclusive boxes plus a torso point), so every
batch, now or later, is labelled the same way and can be added to the others.

Each label has three parts, each doing one job:

| Part | Rule | Gives |
|---|---|---|
| **Box** | The whole fighter, **including the held weapon** | Where the fighter is on screen; the weapon is inside it |
| **Class** | Identity + held weapon | Who it is and what it holds, read from the pixels inside the box |
| **Point** (`torso`) | The centre of the torso | **The position**: it does not move when a weapon is drawn, swung or thrown |

A body-only box could not be drawn consistently: when a weapon covers the body, the visible body
shrinks to the head and the box has to be guessed. A point on the torso can be placed within a few
pixels even then.

## What the labels are for

| The agent needs | Derived from | Labelled here? |
|---|---|---|
| Each fighter's position and velocity | The torso point, frame to frame | **Yes**: the point |
| Who is who | The box's class | **Yes**: `wu_shang_*` / `mordex_*` |
| What each fighter holds (unarmed, or which weapon: different moves, reach and timing) | The box's class | **Yes**: `*_unarmed`, `*_spear`, `*_gauntlets`, `*_scythe` |
| A fighter beyond the screen edge (recovering, about to be KO'd) | Derived: the fighter is not detected and the HUD shows no lost stock | No (decided 2026-11-10; the bubble would add where along the edge it is) |
| Weapons anyone can pick up | Spawn markers | **Yes**: `weapon_spawn` |
| Thrown weapons: the agent's own (only the thrower can pick it up), the bot's (a projectile to avoid) | Thrown-weapon boxes; the owner is tracked, not labelled | **Yes**: `weapon_thrown` |
| Facing, attack startup, stun, the shape of a swing | Fighter crops (the image branch sees them) | No |
| Damage, stocks | HUD reader, checked against the damage numbers (R1) | No |
| Stage edges, blast zones | Fixed geometry, measured once (the camera does not move) | No |

**One model reads all three.** A pose detector (YOLO pose, one keypoint) outputs box, class and
point in one pass. The held weapon is inside the box, so the class is read from the box's own
pixels. A crop classifier on the 256 px window is the fallback if P1 shows the class is not
reliable.

**The crop** given to the agent's image branch is a fixed window of **256 x 256 screen pixels
centred on the torso point**, so it does not shift with the weapon either. It contains the held
weapon and its swing (fighter plus weapon was never wider than 151 px or taller than 174 px in the
old dataset).

**Who owns a thrown weapon** is tracked, not labelled: a `weapon_thrown` box that appears next to
a fighter in the frame that fighter's class changes from armed to unarmed belongs to that fighter,
and keeps that owner while it is followed from frame to frame. The agent's own throws are in the
key log, which checks the tracking for free.

## Classes

| # | Class | What |
|---|---|---|
| 0 | `wu_shang_unarmed` | Wu Shang, empty hands |
| 1 | `wu_shang_spear` | Wu Shang holding his spear |
| 2 | `wu_shang_gauntlets` | Wu Shang wearing his gauntlets |
| 3 | `mordex_unarmed` | Mordex, empty hands |
| 4 | `mordex_scythe` | Mordex holding his scythe |
| 5 | `mordex_gauntlets` | Mordex wearing his gauntlets |
| 6 | `weapon_spawn` | A weapon spawn marker (any legend can pick it up), including while it appears |
| 7 | `weapon_thrown` | A thrown legend weapon, in flight or lying after it landed, whoever threw it |
| 8 | `offscreen` | Reserved, **not labelled**: the bubble at the screen edge that shows a fighter beyond the screen |

## Box rules

**Fighters**

1. **Box: the whole fighter, held weapon included,** as tight as the visible pixels: head, body,
   arms, legs and the weapon in the hands (spear, scythe, gauntlets). **Not** slash arcs, trails,
   hit sparks, dust, dodge afterimages, the KO explosion, the indicator triangle.
2. **The class says what is in the hands in this frame.** Mid-throw, once the weapon has left the
   hands: `*_unarmed`. Picking up, once the weapon is in the hands: armed.
3. **The weapon cannot be seen** (behind the body or an effect): judge from the stance, since armed
   and unarmed poses differ. If you still cannot tell, delete the image.
4. **Point: the centre of the torso,** midway between the bottom of the neck and the belt, on the
   body's middle line; arms, legs and the weapon do not count. Hidden by the weapon, an effect or
   the other fighter: place it where the torso is, judged from the head and legs. Every fighter
   box has its point.
4b. **Behind an effect** (a hit flash, a slash, sparks): box the fighter as if the effect were not
   there; where the effect hides an edge, continue the visible outline.
5. **Overlapping fighters:** one box and one point each, around each whole fighter, including the part the other
   fighter covers; the boxes may overlap as much as the bodies do. A fighter completely hidden by
   the other: no box.
6. **Cut by the screen edge:** box the visible part.
7. **Visible but unusual** (blinking while respawning, mid-dodge, tumbling after a hit, a
   signature move): box it.
8. **Not on screen** (KO'd, waiting to respawn, or beyond the edge with a bubble showing): no box,
   and the bubble is not boxed either. Whether the fighter is beyond the edge or KO'd is told by
   the HUD: no lost stock means it is still alive.

**Weapons**

9. **Spawn marker:** the whole marker, icon and light column, as one box, also while a fighter
   stands in front of it. The marker's arrival is the same class: the red swirl, the fireball and the
   falling flame that come before it (60 of the 470 train spawn boxes are these).
10. **Thrown weapon:** tight around the weapon's own pixels, not its glow or trail; in flight, on
    the stage, or stuck in a wall, until it disappears.
10b. **Weapon points** stay where the import put them, at the box centre.
11. **Never labelled:** the boomerang, bombs, mines and any other item that is not a legend weapon;
    phantoms and other stage decorations.

**Frames**

12. **Complete:** every fighter, bubble and weapon in the frame gets a box. Anything without a box
    is taught as "nothing here".
13. **Not gameplay** (the Settings menu or any other menu open, loading, the KO zoom if the camera moves):
    delete the image. A banner over gameplay (e.g. "SHUTDOWN") is still gameplay.

## Why the old dataset is not reused

The old detector (assets/yolo/best.pt) was trained on the Roboflow project `temporal-yolo` v15
(audited 2026-11-06):

- **The fighter boxes include the held weapon.** A spear stretches the box to twice the body's
  width, so its centre jumps when a weapon is drawn, swung or thrown.
- **Its `weapon` class boxes only the spawn markers.** Thrown weapons have no box.
- **No identity and no held weapon:** one `character` class, plus the indicator triangle.
- 822 unique frames, not 1,860: the training split holds 3 augmented copies of each frame
  (rotation of up to 15 degrees, which the game never shows), all stretched to 960 x 960.
- **All 822 come from one session** (frame_20260114_180307_*), and its train, valid and test
  splits are neighbouring frames of that session, so the old detector's validation scores
  overstate it.

Correcting it would touch nearly every box, which costs as much as correcting new drafts from the
exact setting the agent plays in. The old detector still makes the drafts, and its P1 score on
what it can do (positions and identity) is the baseline.

The "Training Mode" key bar at the top left and the FPS counter are on every frame: they are part
of the setting, not a menu.

## The frames: two sets, from different sessions

Neighbouring frames of one session look alike, so a test frame from a session the detector was
trained on would overstate its score. The two sets never share a session.

| Set | Sessions | Frames | Use |
|---|---|---|---|
| **Test** | one vs Easy, one vs Hard | 75 + 75 | Gate P1 only; never trained on |
| **Train** | four others: two vs Easy, two vs Hard | 100 each, about 400 | Training the detector (Roboflow's valid split is taken from these) |

P1 reports how many test boxes each class has. A class with too few to measure (thrown weapons
and the bubble are rare) gets a batch of frames drafted where it occurs, labelled by these same
rules. More training frames are added the same way, only where the trained detector fails.

The setting never changes: Brawlhaven (small), Wu Shang vs Mordex, **default skins and colours**,
**no training overlays**, 1920 x 1080. Frames recorded otherwise do not belong in either set.

## Workflow

1. Record six sessions of about 10 minutes (`scripts/record_demos.py`), three vs Easy and three vs
   Hard, menu closed before the countdown ends.
2. Draft each set into its own folder:

   ```
   python scripts/prelabel.py <easy 1> <hard 1> --count 75 --out <BRAWL_DATA>/labelling/test
   python scripts/prelabel.py <easy 2> <easy 3> <hard 2> <hard 3> --count 100 --out <BRAWL_DATA>/labelling/train
   ```

   The old detector proposes the boxes (names: `<recording>__<row>.png`). It cannot see held
   weapons or thrown weapons, so **every fighter is drafted as `*_unarmed`**: check the class of
   every fighter box, and draw the thrown weapons.
3. Roboflow: one **object detection** project with the nine classes above. Upload each folder whole
   (images, labels and `data.yaml` together) as its own batch: the test folder goes entirely into
   the **Test** split, the train folder into Train/Valid (about 85/15).
4. Correct every image by the rules above.
5. Generate a version with **no preprocessing** (remove the default resize) and **no
   augmentation** (augmentation is done at training time, without rotation); export **YOLO**
   format and unzip it as `<BRAWL_DATA>/labelling/export/`.

Roboflow renames images (`<name>_png.rf.<hash>.jpg`) and stores them as JPEG; only its label files
are used. They are matched back to the original lossless stills by name, so every label points to
its recording and row (time, keys held).
