# Unattended skill chain: movement (base) -> recovery -> weapon -> spacing -> combat.
#
# Goal: one PPO policy holding every skill, with retention measured after each
# phase. No final agent, no match evaluation -- those come after the chain proves
# retention holds.
#
# Two structural choices for an unattended window:
#
# 1. Recovery is NON-FATAL. It is the only phase that has never produced a usable
#    archive, and it runs first so its demos anchor everything after. If it fails
#    at 00:30 the chain continues without it rather than wasting the night.
#    Every other stage IS fatal: training on an archive that failed validation is
#    worse than not training, and a zero-yield collection looks identical to a
#    good one until something checks.
#
# 2. Recovery gets BC but no PPO. Training episodes start onstage, the offstage
#    setup controller is not wired into training, and the only route to the 2.5
#    success bonus runs through going offstage against a 1.0 death penalty -- so
#    PPO could learn that deliberate self-endangerment pays above ~29% recovery
#    success. Every death measured so far is already a self-destruct.
#
# The weapon classifier is OFF: 95.7% cross-validated, then called unarmed fighters
# ARMED at 1.00 confidence live (trained on 960x960 JPEG exports, served raw
# 1920x1080 captures). Possession falls back to action inference.
#
# Requires: Brawlhalla focused, training mode, hands off the keyboard.
# Budget: ~5.5h of the 8h window, leaving margin for retries.

$ErrorActionPreference = "Stop"
$PY   = "C:\venvs\brawl312\Scripts\python.exe"
$M    = "train\models\easy_bot_v0"
$D    = "train\models\recollected"
$BASE = "$M\llc_movement_fluency.zip"     # movement PPO, already accepted
$STEPS = 200000                            # per PPO phase
$LOG  = "$M\overnight_$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

function Banner($name) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $name"
    Write-Host ("=" * 70)
}

function Run($name, [scriptblock]$block) {
    Banner $name
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

# Validation is a gate, not a report. BC will imitate whatever it is given.
function Validate($path) {
    & $PY tools\validate_llc_demos.py $path --min-samples 100 2>&1 |
        Tee-Object -Append "$LOG\validate.log"
    if ($LASTEXITCODE -ne 0) { throw "demo validation FAILED for $path" }
}

function Collect($phase, $out, $maxEpisodeSteps) {
    & $PY train\collect_bc_locomotion_demos.py --phase $phase `
        --episodes 50 --output $out `
        --max-collection-attempts 200 --min-episode-steps 4 `
        --max-episode-steps $maxEpisodeSteps `
        --no-weapon-classifier --world-replay data\world_replay 2>&1 |
        Tee-Object -Append "$LOG\${phase}_collect.log"
}

function BC($phase, $demos, $resume, $out) {
    & $PY -m train.pretrain_bc_locomotion --phase $phase --demos $demos `
        --resume $resume --epochs 8 --batch-size 512 --learning-rate 0.0002 `
        --entropy-coef 0.001 --goal-relabel --output $out --device cpu 2>&1 |
        Tee-Object -Append "$LOG\${phase}_bc.log"
}

function PPO($phase, $resume, $anchors, $maxEpisodeSteps) {
    & $PY -m train.train_curriculum --phase $phase --resume $resume `
        --timesteps $STEPS --n-steps 512 --batch-size 256 `
        --max-episode-steps $maxEpisodeSteps --bc-demos-path $anchors `
        --save-dir $M --model-name "llc_$phase" `
        --seed 42 --device cpu --log-csv --plot-every 0 2>&1 |
        Tee-Object -Append "$LOG\${phase}_ppo.log"
}

function Retention($phase, $model, $phases, $maxEpisodeSteps) {
    & $PY -m train.evaluate_retention --model $model --phase $phase `
        --phases $phases --episodes 3 --max-episode-steps $maxEpisodeSteps `
        --best-scores $M\retention_best.json `
        --csv "$M\llc_${phase}_retention_eval.csv" --device cpu 2>&1 |
        Tee-Object -Append "$LOG\${phase}_retention.log"
}

$anchors = "$D\movement_fluency_demos.npz"
$current = $BASE
$haveRecovery = $false

try {
    # ---- Recovery: BC from the EXISTING archive, no collection -----------------
    # Not recollected by design. The teacher only produces recovery behaviour after
    # RecoverySetupController has walked the agent off the edge, so another 50
    # episodes would re-sample the same narrow situation. The archive on disk
    # already covers it (2,009 samples, 50 episodes, PASS at last check).
    #
    # Still validated before use: an archive being present is not evidence that it
    # is the one you think it is, and BC will imitate whatever it is handed.
    try {
        Banner "1 recovery BC (existing archive, no collection)"
        Validate "$D\recovery_mastery_demos.npz"
        Run "1 recovery BC" {
            BC "recovery_mastery" "$D\recovery_mastery_demos.npz" $current `
               "$M\llc_recovery_mastery_bc_init.zip"
        }
        $current = "$M\llc_recovery_mastery_bc_init.zip"
        $anchors = "$anchors;$D\recovery_mastery_demos.npz"
        $haveRecovery = $true
    }
    catch {
        Write-Host ""
        Write-Host "RECOVERY SKIPPED: $_" -ForegroundColor Yellow
        Write-Host "Continuing without it. The chain still yields movement/weapon/spacing/combat."
        Add-Content "$LOG\SKIPPED_RECOVERY.txt" "$_"
    }

    # ---- Weapon ---------------------------------------------------------------
    Run "2a weapon collect"  { Collect "weapon_acquisition" "$D\weapon_acquisition_demos.npz" 100 }
    Validate "$D\weapon_acquisition_demos.npz"
    $anchors = "$anchors;$D\weapon_acquisition_demos.npz"
    Run "2b weapon BC"  { BC "weapon_acquisition" "$D\weapon_acquisition_demos.npz" $current "$M\llc_weapon_acquisition_bc_init.zip" }
    Run "2c weapon PPO" { PPO "weapon_acquisition" "$M\llc_weapon_acquisition_bc_init.zip" $anchors 200 }
    $current = "$M\llc_weapon_acquisition.zip"
    $seen = if ($haveRecovery) { "movement_fluency,recovery_mastery,weapon_acquisition" }
            else { "movement_fluency,weapon_acquisition" }
    Run "2d retention" { Retention "weapon_acquisition" $current $seen 200 }

    # ---- Spacing --------------------------------------------------------------
    Run "3a spacing collect"  { Collect "spacing_neutral" "$D\spacing_neutral_demos.npz" 100 }
    Validate "$D\spacing_neutral_demos.npz"
    $anchors = "$anchors;$D\spacing_neutral_demos.npz"
    Run "3b spacing BC"  { BC "spacing_neutral" "$D\spacing_neutral_demos.npz" $current "$M\llc_spacing_neutral_bc_init.zip" }
    Run "3c spacing PPO" { PPO "spacing_neutral" "$M\llc_spacing_neutral_bc_init.zip" $anchors 200 }
    $current = "$M\llc_spacing_neutral.zip"
    $seen = "$seen,spacing_neutral"
    Run "3d retention" { Retention "spacing_neutral" $current $seen 200 }

    # ---- Combat ---------------------------------------------------------------
    Run "4a combat collect"  { Collect "combat_execution" "$D\combat_execution_demos.npz" 600 }
    Validate "$D\combat_execution_demos.npz"
    $anchors = "$anchors;$D\combat_execution_demos.npz"
    Run "4b combat BC"  { BC "combat_execution" "$D\combat_execution_demos.npz" $current "$M\llc_combat_execution_bc_init.zip" }
    Run "4c combat PPO" { PPO "combat_execution" "$M\llc_combat_execution_bc_init.zip" $anchors 600 }
    $current = "$M\llc_combat_execution.zip"
    $seen = "$seen,combat_execution"

    # The paper's Figure 7 bottom row: every skill evaluated by the final policy.
    Run "4d FINAL retention across all skills" { Retention "combat_execution" $current $seen 300 }

    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "COMPLETE at $((Get-Date).ToString('HH:mm:ss'))"
    Write-Host "  policy : $current"
    Write-Host "  matrix : $M\llc_*_retention_eval.csv"
    Write-Host "  recovery included: $haveRecovery"
    Write-Host ("=" * 70)
}
catch {
    Write-Host ""
    Write-Host "STOPPED: $_" -ForegroundColor Red
    Write-Host "Last checkpoint reached: $current"
    Write-Host "Logs in $LOG"
    exit 1
}
