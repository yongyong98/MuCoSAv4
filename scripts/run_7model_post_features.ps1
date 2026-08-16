param(
    [Parameter(Mandatory = $true)]
    [int]$FeatureProcessId
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($env:PAIRBST_PYTHON) { $env:PAIRBST_PYTHON } else { "python" }
$Tag = "official_model_specific_7model_v1"
$FeatureDir = "outputs/runs/features/$Tag"
$ClassificationDir = "outputs/runs/classification/$Tag"
$RetrievalDir = "outputs/runs/retrieval/$Tag"
$StatisticsDir = "outputs/runs/statistics/$Tag"
$FinalDir = "outputs/final_7model_v1"
$LogDir = Join-Path $ProjectRoot "outputs/logs"
$StatusPath = Join-Path $LogDir "post_features_7model_v1.status.json"

$env:PYTHONPATH = (Resolve-Path (Join-Path $ProjectRoot "src")).Path
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONUTF8 = "1"

function Write-Status {
    param([string]$Stage, [string]$Status, [string]$Detail)
    $payload = [ordered]@{
        updated_utc = [DateTime]::UtcNow.ToString("o")
        stage = $Stage
        status = $Status
        detail = $Detail
    }
    [IO.File]::WriteAllText(
        $StatusPath,
        ($payload | ConvertTo-Json -Depth 5),
        [Text.UTF8Encoding]::new($false)
    )
}

function Run-Stage {
    param([string]$Name, [string[]]$Arguments)
    Write-Status -Stage $Name -Status "RUNNING" -Detail "process starting"
    $stdout = Join-Path $LogDir "$Name.stdout.log"
    $stderr = Join-Path $LogDir "$Name.stderr.log"
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        Write-Status -Stage $Name -Status "FAILED" -Detail "exit code $($process.ExitCode)"
        throw "$Name failed with exit code $($process.ExitCode); inspect $stderr"
    }
    Write-Status -Stage $Name -Status "PASS" -Detail "exit code 0"
}

Set-Location $ProjectRoot
Write-Status -Stage "features" -Status "WAITING" -Detail "PID $FeatureProcessId"
while (Get-Process -Id $FeatureProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$featureManifestPath = Join-Path $ProjectRoot "$FeatureDir/extraction_manifest.json"
if (-not (Test-Path -LiteralPath $featureManifestPath)) {
    Write-Status -Stage "features" -Status "FAILED" -Detail "7-model manifest missing"
    throw "Feature process ended without $featureManifestPath"
}
$featureManifest = Get-Content -LiteralPath $featureManifestPath -Raw | ConvertFrom-Json
$expectedModels = @(
    "resnet50_v2", "swin_t", "retccl", "uni", "uni2_h",
    "prov_gigapath", "virchow2"
)
$observedModels = @($featureManifest.models | Sort-Object)
if (($featureManifest.action -ne "features.extract") -or
    ($observedModels.Count -ne 7) -or
    (Compare-Object ($expectedModels | Sort-Object) $observedModels)) {
    Write-Status -Stage "features" -Status "FAILED" -Detail "manifest model grid mismatch"
    throw "Feature manifest does not bind the exact seven-model set"
}
foreach ($model in $expectedModels) {
    $featurePath = Join-Path $ProjectRoot "$FeatureDir/$model.h5"
    if (-not (Test-Path -LiteralPath $featurePath)) {
        throw "Missing completed feature file: $featurePath"
    }
}
Write-Status -Stage "features" -Status "PASS" -Detail "seven completed H5 files and manifest"

$common = @(
    "--config", "configs/paths.local.yaml",
    "--protocol", "configs/protocol_cv3_independent_seed_oof_v1.yaml",
    "--models-config", "configs/models.yaml",
    "--comparisons-config", "configs/comparisons.yaml"
)

Run-Stage -Name "classification_7model_v1" -Arguments (@(
    "-m", "pairbst.cli", "classify", "run"
) + $common + @(
    "--model", "all", "--features-dir", $FeatureDir,
    "--output-dir", $ClassificationDir, "--override-hold"
))

Run-Stage -Name "retrieval_7model_v1" -Arguments (@(
    "-m", "pairbst.cli", "retrieval", "run"
) + $common + @(
    "--model", "all", "--features-dir", $FeatureDir,
    "--output-dir", $RetrievalDir, "--override-hold"
))

Run-Stage -Name "statistics_7model_v1" -Arguments (@(
    "-m", "pairbst.cli", "statistics", "run"
) + $common + @(
    "--classification-dir", $ClassificationDir,
    "--retrieval-dir", $RetrievalDir,
    "--output-dir", $StatisticsDir, "--override-hold"
))

Run-Stage -Name "report_7model_v1" -Arguments (@(
    "-m", "pairbst.cli", "report", "build"
) + $common + @(
    "--classification-dir", $ClassificationDir,
    "--retrieval-dir", $RetrievalDir,
    "--statistics-dir", $StatisticsDir,
    "--output-dir", $FinalDir, "--override-hold"
))

Write-Status -Stage "pipeline" -Status "PASS" -Detail "seven-model versioned report complete"
