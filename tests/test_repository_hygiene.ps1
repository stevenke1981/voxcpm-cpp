[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & git -C $repoRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
    return @($output)
}

$trackedBuildFiles = @(Invoke-Git -Arguments @('ls-files', '--', 'build*/**'))
if ($trackedBuildFiles.Count -ne 0) {
    throw "generated build files are tracked: $($trackedBuildFiles.Count)"
}

foreach ($required in @(
    'THIRD_PARTY_NOTICES.md',
    'scripts/build-release.ps1'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $required))) {
        throw "missing release artifact: $required"
    }
}

foreach ($generatedPath in @(
    'build-cpu/example.obj',
    'build_cuda/example.obj',
    'build-release/example.obj',
    'dist/example.zip',
    'voxcpm-c-windows-x64.zip'
)) {
    & git -C $repoRoot check-ignore --quiet -- $generatedPath
    if ($LASTEXITCODE -ne 0) {
        throw ".gitignore does not cover generated path: $generatedPath"
    }
}

Write-Output 'PASS: repository release hygiene'
