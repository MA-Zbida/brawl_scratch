# Resume the chain after weapon PPO completed (llc_weapon_acquisition.zip, 03:39).
#
# Recovery BC, weapon collect, weapon BC and weapon PPO are all done and preserved.
# This picks up at the weapon retention eval and runs spacing then combat.
#
# The classifier is off throughout: the only trained checkpoint scores a mean
# P(armed) of 0.873 on unarmed live crops. StageGoalEnv now downgrades the
# visual-confirmation requirement automatically when no classifier is attached,
# so no entry point needs to opt out by hand any more.
#
# Budget at the measured 30.5 Hz PPO rate: ~2h per PPO phase, ~4.2h total.

# NOT "Stop". PowerShell wraps a native command's stderr lines in ErrorRecords, so
# with Stop *any* stderr output -- including a plain informational warning -- aborts
# the run even when the process exits 0. Exit codes are the only failure signal here.
$ErrorActionPreference = "Continue"
$PY   = "C:\venvs\brawl312\Scripts\python.exe"
$M    = "train\models\easy_bot_v0"
$D    = "train\models\recollected"
$STEPS = 200000
$LOG  = "$M\resume2_$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

# Native stderr must reach the log or a traceback vanishes, which is how the last
# two failures were invisible until reproduced by hand.
function Run($name, [scriptblock]$block) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $name"
    Write-Host ("=" * 70)
    $global:LASTEXITCODE = 0
    & $block 2>&1 | Tee-Object -Append "$LOG\resume.log"
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

function Validate($path) {
    & $PY scripts\llc\validate_llc_demos.py $path --min-samples 100 2>&1 |
        Tee-Object -Append "$LOG\resume.log"
    if ($LASTEXITCODE -ne 0) { throw "demo validation FAILED for $path" }
}

$anchors = "$D\movement_fluency_demos.npz;$D\recovery_mastery_demos.npz;$D\weapon_acquisition_demos.npz"
$seen    = "movement_fluency,recovery_mastery,weapon_acquisition"

try {
    Run "2d weapon retention" {
        & $PY -m brawl.pipelines.llc.evaluate_retention --model $M\llc_weapon_acquisition.zip `
            --phase weapon_acquisition --phases $seen `
            --episodes 3 --max-episode-steps 200 `
            --best-scores $M\retention_best.json `
            --csv $M\llc_weapon_acquisition_retention_eval.csv --device cpu
    }

    # ---- Spacing ---------------------------------------------------------------
    Run "3a spacing collect" {
        & $PY brawl\pipelines\llc\collect_demos.py --phase spacing_neutral `
            --episodes 50 --output $D\spacing_neutral_demos.npz `
            --max-collection-attempts 200 --min-episode-steps 4 --max-episode-steps 100 `
            --no-weapon-classifier --world-replay data\world_replay
    }
    Validate "$D\spacing_neutral_demos.npz"
    $anchors = "$anchors;$D\spacing_neutral_demos.npz"
    $seen    = "$seen,spacing_neutral"

    Run "3b spacing BC" {
        & $PY -m brawl.pipelines.llc.pretrain_bc --phase spacing_neutral `
            --demos $D\spacing_neutral_demos.npz --resume $M\llc_weapon_acquisition.zip `
            --epochs 8 --batch-size 512 --learning-rate 0.0002 --entropy-coef 0.001 `
            --goal-relabel --output $M\llc_spacing_neutral_bc_init.zip --device cpu
    }
    Run "3c spacing PPO" {
        & $PY -m brawl.pipelines.llc.train_curriculum --phase spacing_neutral `
            --resume $M\llc_spacing_neutral_bc_init.zip `
            --timesteps $STEPS --n-steps 512 --batch-size 256 --max-episode-steps 200 `
            --bc-demos-path $anchors --save-dir $M --model-name llc_spacing_neutral `
            --seed 42 --device cpu --log-csv --plot-every 0
    }
    Run "3d spacing retention" {
        & $PY -m brawl.pipelines.llc.evaluate_retention --model $M\llc_spacing_neutral.zip `
            --phase spacing_neutral --phases $seen --episodes 3 --max-episode-steps 200 `
            --best-scores $M\retention_best.json `
            --csv $M\llc_spacing_neutral_retention_eval.csv --device cpu
    }

    # ---- Combat ----------------------------------------------------------------
    Run "4a combat collect" {
        & $PY brawl\pipelines\llc\collect_demos.py --phase combat_execution `
            --episodes 50 --output $D\combat_execution_demos.npz `
            --max-collection-attempts 200 --min-episode-steps 4 --max-episode-steps 600 `
            --no-weapon-classifier --world-replay data\world_replay
    }
    Validate "$D\combat_execution_demos.npz"
    $anchors = "$anchors;$D\combat_execution_demos.npz"
    $seen    = "$seen,combat_execution"

    Run "4b combat BC" {
        & $PY -m brawl.pipelines.llc.pretrain_bc --phase combat_execution `
            --demos $D\combat_execution_demos.npz --resume $M\llc_spacing_neutral.zip `
            --epochs 8 --batch-size 512 --learning-rate 0.0002 --entropy-coef 0.001 `
            --goal-relabel --output $M\llc_combat_execution_bc_init.zip --device cpu
    }
    Run "4c combat PPO" {
        & $PY -m brawl.pipelines.llc.train_curriculum --phase combat_execution `
            --resume $M\llc_combat_execution_bc_init.zip `
            --timesteps $STEPS --n-steps 512 --batch-size 256 --max-episode-steps 600 `
            --bc-demos-path $anchors --save-dir $M --model-name llc_combat_execution `
            --seed 42 --device cpu --log-csv --plot-every 0
    }
    Run "4d FINAL retention across all skills" {
        & $PY -m brawl.pipelines.llc.evaluate_retention --model $M\llc_combat_execution.zip `
            --phase combat_execution --phases $seen --episodes 3 --max-episode-steps 300 `
            --best-scores $M\retention_best.json `
            --csv $M\llc_combat_execution_retention_eval.csv --device cpu
    }

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
