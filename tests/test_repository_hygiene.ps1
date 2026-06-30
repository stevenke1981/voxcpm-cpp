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
    'LICENSE',
    'THIRD_PARTY_NOTICES.md',
    'scripts/build-release.ps1'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $required))) {
        throw "missing release artifact: $required"
    }
}

$licensePath = Join-Path $repoRoot 'LICENSE'
$licenseText = Get-Content -LiteralPath $licensePath -Raw
foreach ($marker in @(
    'Apache License',
    'Version 2.0, January 2004',
    'http://www.apache.org/licenses/',
    'END OF TERMS AND CONDITIONS'
)) {
    if (-not $licenseText.Contains($marker)) {
        throw "LICENSE is not the complete Apache License 2.0 text: missing '$marker'"
    }
}

$releaseScript = Get-Content -LiteralPath (
    Join-Path $repoRoot 'scripts\build-release.ps1') -Raw
if ($releaseScript -notmatch "'LICENSE'") {
    throw 'release package does not include LICENSE'
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
