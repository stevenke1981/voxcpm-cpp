$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$validator = Join-Path $root 'scripts\validate-backend-matrix.ps1'
$temp = Join-Path ([IO.Path]::GetTempPath()) "voxcpm-backend-report-$PID.json"

$valid = @(
    [ordered]@{
        name = 'cpu-f16'
        backend = 'cpu'
        model_format = 'f16'
        status = 'passed'
        exit_code = 0
        wall_ms = 1000.0
        peak_working_set_mb = 512.0
        audio_sec = 1.0
        rtf = 1.0
        sample_rate = 48000
        n_samples = 48000
        finite_samples = 48000
        peak = 0.5
        rms = 0.1
        clipped_samples = 0
    }
)

try {
    $valid | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temp -Encoding utf8
    & $validator -ValidateReport $temp
    if ($LASTEXITCODE -ne 0) {
        throw "valid backend report was rejected"
    }

    [ordered]@{ name = 'broken' } |
        ConvertTo-Json |
        Set-Content -LiteralPath $temp -Encoding utf8
    & $validator -ValidateReport $temp
    if ($LASTEXITCODE -eq 0) {
        throw "invalid backend report was accepted"
    }
} finally {
    Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
}

Write-Host 'PASS: backend matrix report schema'
