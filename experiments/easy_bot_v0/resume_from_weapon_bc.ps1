# Resume the overnight chain from weapon BC.
#
# Recovery BC and weapon collection already succeeded:
#   train\models\easy_bot_v0\llc_recovery_mastery_bc_init.zip
#   train\models\recollected\weapon_acquisition_demos.npz   (2,817 samples, PASS)
# Neither is redone here.
#
# The original run died in weapon BC on a guard added for the *collector*: a stage
# requiring visual weapon confirmation refuses to build without a classifier. BC
# builds a StageGoalEnv only to read the observation and action spaces and never
# steps it, so that guard was wrong there and is now cleared for BC.
#
# The more consequential fix: train_curriculum hardcoded weapon_classifier_enabled
# =True, so weapon PPO would have trained against the classifier we measured at a
# mean P(armed) of 0.873 on unarmed live crops -- corrupting player_has_weapon and
# every retention number, while looking like a clean run. It is now off by default
# and passed explicitly below.

$ErrorActionPreference = "Stop"
$PY   = "C:\venvs\brawl312\Scripts\python.exe"
$M    = "train\models\easy_bot_v0"
$D    = "train\models\recollected"
$STEPS = 200000
$LOG  = "$M\resume_$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

function Run($name, [scriptblock]$block) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $name"
    Write-Host ("=" * 70)
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

function Validate($path) {
    & $PY tools\validate_llc_demos.py $path --min-samples 100
    if ($LASTEXITCODE -ne 0) { throw "demo validation FAILED for $path" }
}

function Collect($phase, $out, $maxEpisodeSteps) {
    & $PY train\collect_bc_locomotion_demos.py --phase $phase `
        --episodes 50 --output $out `
        --max-collection-attempts 200 --min-episode-steps 4 `
        --max-episode-steps $maxEpisodeSteps `
        --no-weapon-classifier --world-replay data\world_replay
}

function BC($phase, $demos, $resume, $out) {
    & $PY -m train.pretrain_bc_locomotion --phase $phase --demos $demos `
        --resume $resume --epochs 8 --batch-size 512 --learning-rate 0.0002 `
        --entropy-coef 0.001 --goal-relabel --output $out --device cpu
}

function PPO($phase, $resume, $anchors, $maxEpisodeSteps) {
    & $PY -m train.train_curriculum --phase $phase --resume $resume `
        --timesteps $STEPS --n-steps 512 --batch-size 256 `
        --max-episode-steps $maxEpisodeSteps --bc-demos-path $anchors `
        --save-dir $M --model-name "llc_$phase" `
        --no-weapon-classifier `
        --seed 42 --device cpu --log-csv --plot-every 0
}

function Retention($phase, $model, $phases, $maxEpisodeSteps) {
    & $PY -m train.evaluate_retention --model $model --phase $phase `
        --phases $phases --episodes 3 --max-episode-steps $maxEpisodeSteps `
        --best-scores $M\retention_best.json `
        --csv "$M\llc_${phase}_retention_eval.csv" --device cpu
}

$anchors = "$D\movement_fluency_demos.npz;$D\recovery_mastery_demos.npz;$D\weapon_acquisition_demos.npz"
$seen    = "movement_fluency,recovery_mastery,weapon_acquisition"

try {
    Start-Transcript -Path "$LOG\resume.log" -Append | Out-Null

    Run "2b weapon BC"  { BC "weapon_acquisition" "$D\weapon_acquisition_demos.npz" `
                             "$M\llc_recovery_mastery_bc_init.zip" `
                             "$M\llc_weapon_acquisition_bc_init.zip" }
    Run "2c weapon PPO" { PPO "weapon_acquisition" "$M\llc_weapon_acquisition_bc_init.zip" $anchors 200 }
    Run "2d retention"  { Retention "weapon_acquisition" "$M\llc_weapon_acquisition.zip" $seen 200 }

    Run "3a spacing collect" { Collect "spacing_neutral" "$D\spacing_neutral_demos.npz" 100 }
    Validate "$D\spacing_neutral_demos.npz"
    $anchors = "$anchors;$D\spacing_neutral_demos.npz"
    $seen    = "$seen,spacing_neutral"
    Run "3b spacing BC"  { BC "spacing_neutral" "$D\spacing_neutral_demos.npz" "$M\llc_weapon_acquisition.zip" "$M\llc_spacing_neutral_bc_init.zip" }
    Run "3c spacing PPO" { PPO "spacing_neutral" "$M\llc_spacing_neutral_bc_init.zip" $anchors 200 }
    Run "3d retention"   { Retention "spacing_neutral" "$M\llc_spacing_neutral.zip" $seen 200 }

    Run "4a combat collect" { Collect "combat_execution" "$D\combat_execution_demos.npz" 600 }
    Validate "$D\combat_execution_demos.npz"
    $anchors = "$anchors;$D\combat_execution_demos.npz"
    $seen    = "$seen,combat_execution"
    Run "4b combat BC"  { BC "combat_execution" "$D\combat_execution_demos.npz" "$M\llc_spacing_neutral.zip" "$M\llc_combat_execution_bc_init.zip" }
    Run "4c combat PPO" { PPO "combat_execution" "$M\llc_combat_execution_bc_init.zip" $anchors 600 }
    Run "4d FINAL retention across all skills" { Retention "combat_execution" "$M\llc_combat_execution.zip" $seen 300 }

    Write-Host ""
    Write-Host "COMPLETE at $((Get-Date).ToString('HH:mm:ss'))"
    Write-Host "  policy : $M\llc_combat_execution.zip"
    Write-Host "  matrix : $M\llc_*_retention_eval.csv"
}
catch {
    Write-Host ""
    Write-Host "STOPPED: $_" -ForegroundColor Red
    Write-Host "Full output including tracebacks: $LOG\resume.log"
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
