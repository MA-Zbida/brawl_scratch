param(
    [string]$From = "",
    [string]$To = "",
    [ValidateRange(0, 2147483647)]
    [int]$Timesteps = 0,
    [switch]$Smoke,
    [switch]$DryRun,
    [ValidateSet("", "warning", "failure", "artifacts", "recovery")]
    [string]$HarnessProbe = "",
    [string]$StateRoot = "",
    [string]$LogRoot = ""
)

# Native programs may legitimately write warnings to stderr while exiting zero.
# PowerShell turns those lines into ErrorRecords, so Stop would kill a healthy run.
$ErrorActionPreference = "Continue"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot "..\..")).Path
$Python = "C:\venvs\brawl312\Scripts\python.exe"
$ProductionModels = Join-Path $RepoRoot "train\models\easy_bot_v0"
$ProductionDemos = Join-Path $RepoRoot "train\models\recollected"

if ([string]::IsNullOrWhiteSpace($StateRoot)) {
    $StateRoot = Join-Path $ScriptRoot ".chain_state"
}
if ([string]::IsNullOrWhiteSpace($LogRoot)) {
    $LogRoot = Join-Path $ScriptRoot "logs"
}

$Mode = if ($Smoke) { "smoke" } else { "production" }
$StateRoot = Join-Path $StateRoot $Mode
New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Utf8 = New-Object System.Text.UTF8Encoding($false)
if ($HarnessProbe) {
    $LogFile = Join-Path $LogRoot "harness.log"
} else {
    $LogFile = Join-Path $LogRoot ("chain_{0}_{1}.log" -f $Mode, (Get-Date -Format "yyyyMMdd-HHmmss"))
}
[System.IO.File]::WriteAllText($LogFile, "", $Utf8)

function Write-LogLine([string]$Text) {
    [Console]::Out.WriteLine($Text)
    [System.IO.File]::AppendAllText($LogFile, $Text + [Environment]::NewLine, $Utf8)
}

function ConvertTo-NativeArgument([string]$Value) {
    # Every current argument is a path, number, or identifier. Quoting all of them
    # keeps semicolon-separated BC paths intact and makes spaces in temp paths safe.
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-NativeStage([string]$Stage, [string[]]$Arguments) {
    Write-LogLine ""
    Write-LogLine ("=" * 72)
    Write-LogLine ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Stage)
    Write-LogLine ("=" * 72)

    # A previous non-zero native exit must never poison the next command.
    $global:LASTEXITCODE = 0

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Python
    $startInfo.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join " ")
    $startInfo.WorkingDirectory = $RepoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = $Utf8
    $startInfo.StandardErrorEncoding = $Utf8

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "process did not start"
        }
        # Read both pipes asynchronously; sequential ReadToEnd can deadlock when a
        # traceback or progress renderer fills the other redirected pipe.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        $global:LASTEXITCODE = [int]$process.ExitCode

        if (-not [string]::IsNullOrEmpty($stdout)) {
            [Console]::Out.Write($stdout)
            [System.IO.File]::AppendAllText($LogFile, $stdout, $Utf8)
        }
        if (-not [string]::IsNullOrEmpty($stderr)) {
            [Console]::Error.Write($stderr)
            [System.IO.File]::AppendAllText($LogFile, $stderr, $Utf8)
        }
        if ($global:LASTEXITCODE -ne 0) {
            $message = "[$Stage] failed with exit code $global:LASTEXITCODE"
            Write-LogLine $message
            throw $message
        }
    }
    finally {
        $process.Dispose()
    }
}

function Get-MarkerPath([string]$Stage) {
    return Join-Path $StateRoot ($Stage + ".done.json")
}

function Assert-Artifacts([string]$Stage, [string[]]$Artifacts) {
    foreach ($artifact in $Artifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "[$Stage] exited zero but required artifact is missing: $artifact"
        }
    }
}

function Complete-Stage([string]$Stage, [string[]]$Artifacts, [string]$Source = "executed") {
    Assert-Artifacts $Stage $Artifacts
    $payload = [ordered]@{
        stage = $Stage
        source = $Source
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        artifacts = @($Artifacts | ForEach-Object { [System.IO.Path]::GetFullPath($_) })
    } | ConvertTo-Json -Depth 3
    [System.IO.File]::WriteAllText((Get-MarkerPath $Stage), $payload + [Environment]::NewLine, $Utf8)
}

function Test-CompletedStage([string]$Stage, [string[]]$Artifacts) {
    $marker = Get-MarkerPath $Stage
    if (-not (Test-Path -LiteralPath $marker)) {
        return $false
    }
    Assert-Artifacts $Stage $Artifacts
    return $true
}

function Test-RecoverableArtifacts([string]$DependencyStage, [string[]]$Artifacts) {
    $dependencyMarker = Get-MarkerPath $DependencyStage
    if (-not (Test-Path -LiteralPath $dependencyMarker)) {
        return $false
    }
    $dependencyTime = (Get-Item -LiteralPath $dependencyMarker).LastWriteTimeUtc
    foreach ($artifact in $Artifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            return $false
        }
        if ((Get-Item -LiteralPath $artifact).LastWriteTimeUtc -lt $dependencyTime) {
            return $false
        }
    }
    return $true
}

$StageOrder = @(
    "recovery_bc",
    "weapon_collect",
    "weapon_bc",
    "weapon_ppo",
    "weapon_retention",
    "spacing_collect",
    "spacing_bc",
    "spacing_ppo",
    "spacing_retention",
    "combat_collect",
    "combat_bc",
    "combat_ppo",
    "combat_retention"
)

function Resolve-FromStage([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $StageOrder[0] }
    $normalized = $Value.Trim().ToLowerInvariant().Replace("-", "_")
    $aliases = @{
        recovery = "recovery_bc"
        weapon = "weapon_collect"
        spacing = "spacing_collect"
        combat = "combat_collect"
    }
    if ($aliases.ContainsKey($normalized)) { $normalized = $aliases[$normalized] }
    if ($StageOrder -notcontains $normalized) {
        throw "Unknown --from stage '$Value'. Expected recovery, weapon, spacing, combat, or an exact stage name."
    }
    return $normalized
}

function Resolve-ToStage([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $StageOrder[$StageOrder.Count - 1] }
    $normalized = $Value.Trim().ToLowerInvariant().Replace("-", "_")
    $aliases = @{
        recovery = "recovery_bc"
        weapon = "weapon_retention"
        spacing = "spacing_retention"
        combat = "combat_retention"
    }
    if ($aliases.ContainsKey($normalized)) { $normalized = $aliases[$normalized] }
    if ($StageOrder -notcontains $normalized) {
        throw "Unknown --to stage '$Value'. Expected recovery, weapon, spacing, combat, or an exact stage name."
    }
    return $normalized
}

$FromStage = Resolve-FromStage $From
$FromIndex = [Array]::IndexOf($StageOrder, $FromStage)
$ToStage = Resolve-ToStage $To
$ToIndex = [Array]::IndexOf($StageOrder, $ToStage)
if ($FromIndex -gt $ToIndex) {
    throw "--from '$FromStage' comes after --to '$ToStage'; choose a forward stage range."
}

if ($Smoke) {
    $Episodes = 2
    $BcEpochs = 1
    $PpoSteps = 512
    $RetentionEpisodes = 1
    $Attempts = 12
    $PlotEvery = 1
    $ModelDir = Join-Path $ProductionModels "smoke"
    $DemoDir = Join-Path $ModelDir "demos"
} else {
    $Episodes = 50
    $BcEpochs = 8
    $PpoSteps = 200000
    $RetentionEpisodes = 3
    $Attempts = 200
    $PlotEvery = 20
    $ModelDir = $ProductionModels
    $DemoDir = $ProductionDemos
}
if ($Timesteps -gt 0) {
    $PpoSteps = $Timesteps
}
$MovementDemo = Join-Path $ProductionDemos "movement_fluency_demos.npz"
$RecoveryDemo = Join-Path $ProductionDemos "recovery_mastery_demos.npz"
$WeaponDemo = Join-Path $DemoDir "weapon_acquisition_demos.npz"
$SpacingDemo = Join-Path $DemoDir "spacing_neutral_demos.npz"
$CombatDemo = Join-Path $DemoDir "combat_execution_demos.npz"
$MovementModel = Join-Path $ProductionModels "llc_movement_fluency.zip"
$RecoveryModel = Join-Path $ModelDir "llc_recovery_mastery_bc_init.zip"
$WeaponBcModel = Join-Path $ModelDir "llc_weapon_acquisition_bc_init.zip"
$WeaponModel = Join-Path $ModelDir "llc_weapon_acquisition.zip"
$SpacingBcModel = Join-Path $ModelDir "llc_spacing_neutral_bc_init.zip"
$SpacingModel = Join-Path $ModelDir "llc_spacing_neutral.zip"
$CombatBcModel = Join-Path $ModelDir "llc_combat_execution_bc_init.zip"
$CombatModel = Join-Path $ModelDir "llc_combat_execution.zip"
$RecoveryArtifacts = @($RecoveryModel, "${RecoveryModel}.metadata.json")
$WeaponBcArtifacts = @($WeaponBcModel, "${WeaponBcModel}.metadata.json")
$WeaponArtifacts = @($WeaponModel, "${WeaponModel}.metadata.json")
$SpacingBcArtifacts = @($SpacingBcModel, "${SpacingBcModel}.metadata.json")
$SpacingArtifacts = @($SpacingModel, "${SpacingModel}.metadata.json")
$CombatBcArtifacts = @($CombatBcModel, "${CombatBcModel}.metadata.json")
$CombatArtifacts = @($CombatModel, "${CombatModel}.metadata.json")
$BestScores = Join-Path $ModelDir "retention_best.json"
$WorldReplay = Join-Path $RepoRoot "data\world_replay"

Write-LogLine ("MODE={0} FROM={1} TO={2}" -f $Mode, $FromStage, $ToStage)
Write-LogLine ("[BUDGET] episodes={0} bc_epochs={1} ppo_steps={2} retention_episodes={3}" -f $Episodes, $BcEpochs, $PpoSteps, $RetentionEpisodes)

if ($HarnessProbe) {
    try {
        if ($HarnessProbe -eq "recovery") {
            $dependencyArtifact = Join-Path $StateRoot "dependency.txt"
            [System.IO.File]::WriteAllText($dependencyArtifact, "dependency", $Utf8)
            Complete-Stage "dependency_stage" @($dependencyArtifact) "harness"
            $dependencyTime = (Get-Item -LiteralPath (Get-MarkerPath "dependency_stage")).LastWriteTimeUtc

            $staleModel = Join-Path $StateRoot "stale.zip"
            $staleMetadata = "${staleModel}.metadata.json"
            [System.IO.File]::WriteAllText($staleModel, "stale", $Utf8)
            [System.IO.File]::WriteAllText($staleMetadata, "stale", $Utf8)
            [System.IO.File]::SetLastWriteTimeUtc($staleModel, $dependencyTime.AddSeconds(-1))
            [System.IO.File]::SetLastWriteTimeUtc($staleMetadata, $dependencyTime.AddSeconds(-1))
            if (Test-RecoverableArtifacts "dependency_stage" @($staleModel, $staleMetadata)) {
                throw "stale outputs were incorrectly recoverable"
            }

            $freshModel = Join-Path $StateRoot "fresh.zip"
            $freshMetadata = "${freshModel}.metadata.json"
            [System.IO.File]::WriteAllText($freshModel, "fresh", $Utf8)
            [System.IO.File]::WriteAllText($freshMetadata, "fresh", $Utf8)
            [System.IO.File]::SetLastWriteTimeUtc($freshModel, $dependencyTime.AddSeconds(1))
            [System.IO.File]::SetLastWriteTimeUtc($freshMetadata, $dependencyTime.AddSeconds(1))
            $freshArtifacts = @($freshModel, $freshMetadata)
            if (-not (Test-RecoverableArtifacts "dependency_stage" $freshArtifacts)) {
                throw "fresh outputs were not recoverable"
            }
            Complete-Stage "recovered_stage" $freshArtifacts "recovered_after_postcondition_failure"
            Write-LogLine "HARNESS_RECOVERY_READY"
            exit 0
        }
        if ($HarnessProbe -eq "artifacts") {
            $probeModel = Join-Path $StateRoot "probe.zip"
            $probeMetadata = "${probeModel}.metadata.json"
            [System.IO.File]::WriteAllText($probeModel, "probe", $Utf8)
            [System.IO.File]::WriteAllText($probeMetadata, "metadata", $Utf8)
            $probeArtifacts = @($probeModel, $probeMetadata)
            Complete-Stage "artifact_binding" $probeArtifacts
            Write-LogLine "HARNESS_ARTIFACTS_RECORDED"
            exit 0
        }
        if ($HarnessProbe -eq "warning") {
            Invoke-NativeStage "fault_warning" @(
                "-c",
                "import sys; print('PROBE_WARNING_ON_STDERR', file=sys.stderr); print('probe stdout')"
            )
            Invoke-NativeStage "after_warning" @("-c", "print('HARNESS_WARNING_CONTINUED')")
            exit 0
        }
        Invoke-NativeStage "fault_nonzero" @(
            "-c",
            "import sys; print('TRACEBACK_SENTINEL', file=sys.stderr); raise RuntimeError('injected failure')"
        )
        throw "fault_nonzero unexpectedly returned zero"
    }
    catch {
        Write-LogLine ("STOPPED: " + $_.Exception.Message)
        [Console]::Error.WriteLine("STOPPED: " + $_.Exception.Message)
        exit 1
    }
}

if ($DryRun) {
    Write-LogLine "[PLAN] preflight"
    for ($index = $FromIndex; $index -le $ToIndex; $index++) {
        Write-LogLine ("[PLAN] " + $StageOrder[$index])
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $DemoDir | Out-Null

function Invoke-PipelineStage(
    [string]$Stage,
    [string[]]$Artifacts,
    [scriptblock]$Action,
    [switch]$ImportExisting,
    [string]$RecoverExistingAfter = ""
) {
    $index = [Array]::IndexOf($StageOrder, $Stage)
    if ($index -lt $FromIndex) {
        Write-LogLine "[SKIP before --from] $Stage"
        return
    }
    if ($index -gt $ToIndex) {
        Write-LogLine "[SKIP after --to] $Stage"
        return
    }
    if (Test-CompletedStage $Stage $Artifacts) {
        Write-LogLine "[SKIP marker] $Stage"
        return
    }
    if ($ImportExisting -and -not $Smoke) {
        $allExist = $true
        foreach ($artifact in $Artifacts) {
            if (-not (Test-Path -LiteralPath $artifact)) { $allExist = $false }
        }
        if ($allExist) {
            Complete-Stage $Stage $Artifacts "preserved_existing_artifact"
            Write-LogLine "[SKIP preserved] $Stage"
            return
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($RecoverExistingAfter)) {
        if (Test-RecoverableArtifacts $RecoverExistingAfter $Artifacts) {
            Complete-Stage $Stage $Artifacts "recovered_after_postcondition_failure"
            Write-LogLine "[RECOVERED existing output] $Stage"
            return
        }
    }
    & $Action
    Complete-Stage $Stage $Artifacts
    Write-LogLine "[DONE] $Stage"
}

try {
    # Preflight deliberately runs on every invocation. A stale marker is not proof
    # that today's capture, model, and archive lifecycle still works.
    Invoke-NativeStage "preflight" @("scripts\llc\preflight_pipeline.py")

    # Recovery is BC-only from the existing validated archive. There is no
    # recovery_collect stage: its heuristic only becomes useful once setup has
    # already moved the fighter offstage, so another collection pass buys nothing.
    Invoke-PipelineStage "recovery_bc" $RecoveryArtifacts {
        Invoke-NativeStage "recovery_bc" @(
            "-m", "brawl.pipelines.llc.pretrain_bc",
            "--phase", "recovery_mastery",
            "--demos", $RecoveryDemo,
            "--resume", $MovementModel,
            "--epochs", "$BcEpochs",
            "--batch-size", "512",
            "--learning-rate", "0.0002",
            "--entropy-coef", "0.001",
            "--goal-relabel",
            "--output", $RecoveryModel,
            "--device", "cpu"
        )
    } -ImportExisting

    Invoke-PipelineStage "weapon_collect" @($WeaponDemo) {
        Invoke-NativeStage "weapon_collect" @(
            "brawl\pipelines\llc\collect_demos.py",
            "--phase", "weapon_acquisition",
            "--episodes", "$Episodes",
            "--output", $WeaponDemo,
            "--max-collection-attempts", "$Attempts",
            "--min-episode-steps", "4",
            "--max-episode-steps", "100",
            "--no-weapon-classifier",
            "--world-replay", $WorldReplay
        )
        Invoke-NativeStage "weapon_demo_validation" @(
            "scripts\llc\validate_llc_demos.py", $WeaponDemo,
            "--phase", "weapon_acquisition", "--min-samples", "1"
        )
    } -ImportExisting

    Invoke-PipelineStage "weapon_bc" $WeaponBcArtifacts {
        Invoke-NativeStage "weapon_bc" @(
            "-m", "brawl.pipelines.llc.pretrain_bc",
            "--phase", "weapon_acquisition",
            "--demos", $WeaponDemo,
            "--resume", $RecoveryModel,
            "--epochs", "$BcEpochs",
            "--batch-size", "512",
            "--learning-rate", "0.0002",
            "--entropy-coef", "0.001",
            "--goal-relabel",
            "--output", $WeaponBcModel,
            "--device", "cpu"
        )
    } -ImportExisting

    $WeaponAnchors = "$MovementDemo;$RecoveryDemo;$WeaponDemo"
    Invoke-PipelineStage "weapon_ppo" $WeaponArtifacts {
        Invoke-NativeStage "weapon_ppo" @(
            "-m", "brawl.pipelines.llc.train_curriculum",
            "--phase", "weapon_acquisition",
            "--resume", $WeaponBcModel,
            "--timesteps", "$PpoSteps",
            "--n-steps", "512",
            "--batch-size", "256",
            "--max-episode-steps", "200",
            "--bc-demos-path", $WeaponAnchors,
            "--save-dir", $ModelDir,
            "--model-name", "llc_weapon_acquisition",
            "--seed", "42", "--device", "cpu",
            "--no-weapon-classifier", "--log-csv",
            "--plot-every", "$PlotEvery"
        )
    } -ImportExisting

    $WeaponRetention = Join-Path $ModelDir "llc_weapon_acquisition_retention_eval.csv"
    Invoke-PipelineStage "weapon_retention" @($WeaponRetention) {
        Invoke-NativeStage "weapon_retention" @(
            "-m", "brawl.pipelines.llc.evaluate_retention",
            "--model", $WeaponModel,
            "--phase", "weapon_acquisition",
            "--phases", "movement_fluency,recovery_mastery,weapon_acquisition",
            "--episodes", "$RetentionEpisodes",
            "--max-episode-steps", "200",
            "--best-scores", $BestScores,
            "--csv", $WeaponRetention,
            "--device", "cpu"
        )
    }

    Invoke-PipelineStage "spacing_collect" @($SpacingDemo) {
        Invoke-NativeStage "spacing_collect" @(
            "brawl\pipelines\llc\collect_demos.py",
            "--phase", "spacing_neutral",
            "--episodes", "$Episodes",
            "--output", $SpacingDemo,
            "--max-collection-attempts", "$Attempts",
            "--min-episode-steps", "4",
            "--max-episode-steps", "100",
            "--no-weapon-classifier",
            "--world-replay", $WorldReplay
        )
        Invoke-NativeStage "spacing_demo_validation" @(
            "scripts\llc\validate_llc_demos.py", $SpacingDemo,
            "--phase", "spacing_neutral", "--min-samples", "1"
        )
    }

    Invoke-PipelineStage "spacing_bc" $SpacingBcArtifacts {
        Invoke-NativeStage "spacing_bc" @(
            "-m", "brawl.pipelines.llc.pretrain_bc",
            "--phase", "spacing_neutral",
            "--demos", $SpacingDemo,
            "--resume", $WeaponModel,
            "--epochs", "$BcEpochs",
            "--batch-size", "512",
            "--learning-rate", "0.0002",
            "--entropy-coef", "0.001",
            "--goal-relabel",
            "--output", $SpacingBcModel,
            "--device", "cpu"
        )
    } -RecoverExistingAfter "spacing_collect"

    $SpacingAnchors = "$WeaponAnchors;$SpacingDemo"
    Invoke-PipelineStage "spacing_ppo" $SpacingArtifacts {
        Invoke-NativeStage "spacing_ppo" @(
            "-m", "brawl.pipelines.llc.train_curriculum",
            "--phase", "spacing_neutral",
            "--resume", $SpacingBcModel,
            "--timesteps", "$PpoSteps",
            "--n-steps", "512", "--batch-size", "256",
            "--max-episode-steps", "200",
            "--bc-demos-path", $SpacingAnchors,
            "--save-dir", $ModelDir, "--model-name", "llc_spacing_neutral",
            "--seed", "42", "--device", "cpu",
            "--no-weapon-classifier", "--log-csv",
            "--plot-every", "$PlotEvery"
        )
    } -RecoverExistingAfter "spacing_bc"

    $SpacingRetention = Join-Path $ModelDir "llc_spacing_neutral_retention_eval.csv"
    Invoke-PipelineStage "spacing_retention" @($SpacingRetention) {
        Invoke-NativeStage "spacing_retention" @(
            "-m", "brawl.pipelines.llc.evaluate_retention",
            "--model", $SpacingModel, "--phase", "spacing_neutral",
            "--phases", "movement_fluency,recovery_mastery,weapon_acquisition,spacing_neutral",
            "--episodes", "$RetentionEpisodes", "--max-episode-steps", "200",
            "--best-scores", $BestScores, "--csv", $SpacingRetention,
            "--device", "cpu"
        )
    }

    Invoke-PipelineStage "combat_collect" @($CombatDemo) {
        Invoke-NativeStage "combat_collect" @(
            "brawl\pipelines\llc\collect_demos.py",
            "--phase", "combat_execution", "--episodes", "$Episodes",
            "--output", $CombatDemo, "--max-collection-attempts", "$Attempts",
            "--min-episode-steps", "4", "--max-episode-steps", "600",
            "--no-weapon-classifier", "--world-replay", $WorldReplay
        )
        Invoke-NativeStage "combat_demo_validation" @(
            "scripts\llc\validate_llc_demos.py", $CombatDemo,
            "--phase", "combat_execution", "--min-samples", "1"
        )
    }

    Invoke-PipelineStage "combat_bc" $CombatBcArtifacts {
        Invoke-NativeStage "combat_bc" @(
            "-m", "brawl.pipelines.llc.pretrain_bc",
            "--phase", "combat_execution", "--demos", $CombatDemo,
            "--resume", $SpacingModel, "--epochs", "$BcEpochs",
            "--batch-size", "512", "--learning-rate", "0.0002",
            "--entropy-coef", "0.001", "--goal-relabel",
            "--output", $CombatBcModel, "--device", "cpu"
        )
    } -RecoverExistingAfter "combat_collect"

    $CombatAnchors = "$SpacingAnchors;$CombatDemo"
    Invoke-PipelineStage "combat_ppo" $CombatArtifacts {
        Invoke-NativeStage "combat_ppo" @(
            "-m", "brawl.pipelines.llc.train_curriculum",
            "--phase", "combat_execution", "--resume", $CombatBcModel,
            "--timesteps", "$PpoSteps", "--n-steps", "512",
            "--batch-size", "256", "--max-episode-steps", "600",
            "--bc-demos-path", $CombatAnchors,
            "--save-dir", $ModelDir, "--model-name", "llc_combat_execution",
            "--seed", "42", "--device", "cpu",
            "--no-weapon-classifier", "--log-csv",
            "--plot-every", "$PlotEvery"
        )
    } -RecoverExistingAfter "combat_bc"

    $CombatRetention = Join-Path $ModelDir "llc_combat_execution_retention_eval.csv"
    Invoke-PipelineStage "combat_retention" @($CombatRetention) {
        Invoke-NativeStage "combat_retention" @(
            "-m", "brawl.pipelines.llc.evaluate_retention", "--model", $CombatModel,
            "--phase", "combat_execution",
            "--phases", "movement_fluency,recovery_mastery,weapon_acquisition,spacing_neutral,combat_execution",
            "--episodes", "$RetentionEpisodes", "--max-episode-steps", "300",
            "--best-scores", $BestScores, "--csv", $CombatRetention,
            "--device", "cpu"
        )
    }

    Write-LogLine ""
    Write-LogLine ("REQUESTED RANGE COMPLETE: {0} -> {1}" -f $FromStage, $ToStage)
    if ($ToIndex -ge [Array]::IndexOf($StageOrder, "combat_ppo")) {
        Write-LogLine ("latest checkpoint: " + $CombatModel)
    } elseif ($ToIndex -ge [Array]::IndexOf($StageOrder, "spacing_ppo")) {
        Write-LogLine ("latest checkpoint: " + $SpacingModel)
    } elseif ($ToIndex -ge [Array]::IndexOf($StageOrder, "weapon_ppo")) {
        Write-LogLine ("latest checkpoint: " + $WeaponModel)
    } elseif ($ToIndex -ge [Array]::IndexOf($StageOrder, "recovery_bc")) {
        Write-LogLine ("latest checkpoint: " + $RecoveryModel)
    }
    Write-LogLine ("full log: " + $LogFile)
    exit 0
}
catch {
    $message = "STOPPED: " + $_.Exception.Message
    Write-LogLine $message
    [Console]::Error.WriteLine($message)
    Write-LogLine ("Full stdout/stderr log: " + $LogFile)
    exit 1
}
