[CmdletBinding()]
param(
    [string]$BuildDir = 'build-release',
    [string]$OutputDir = 'dist',
    [string]$Model = '',
    [switch]$SkipTests,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Resolve-RepositoryPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $candidate = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
    }
    $prefix = $repoRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must stay inside the repository: $candidate"
    }
    return $candidate
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "> $Command $($Arguments -join ' ')"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

$buildPath = Resolve-RepositoryPath -Path $BuildDir -Description 'BuildDir'
$outputPath = Resolve-RepositoryPath -Path $OutputDir -Description 'OutputDir'
if ($buildPath -eq $repoRoot -or $outputPath -eq $repoRoot) {
    throw 'BuildDir and OutputDir cannot be the repository root'
}

$hygieneTest = Join-Path $repoRoot 'tests\test_repository_hygiene.ps1'
& $hygieneTest

$modelPath = ''
if ($Model) {
    $modelPath = if ([IO.Path]::IsPathRooted($Model)) {
        [IO.Path]::GetFullPath($Model)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $Model))
    }
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
        throw "Model does not exist: $modelPath"
    }
    if ([IO.Path]::GetExtension($modelPath) -ne '.gguf') {
        throw "Model must be a GGUF file: $modelPath"
    }
}

if ($Clean) {
    foreach ($path in @($buildPath, $outputPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
if ($modelPath) {
    $env:VCPM_MODEL = $modelPath
} else {
    Remove-Item Env:VCPM_MODEL -ErrorAction SilentlyContinue
}

Invoke-Checked -Command 'cmake' -Arguments @(
    '-S', $repoRoot,
    '-B', $buildPath,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DVCPM_BUILD_TESTS=ON',
    '-DVCPM_GGML_FETCH=ON'
)
Invoke-Checked -Command 'cmake' -Arguments @(
    '--build', $buildPath,
    '--config', 'Release',
    '-j', [Environment]::ProcessorCount
)

if (-not $SkipTests) {
    Invoke-Checked -Command 'ctest' -Arguments @(
        '--test-dir', $buildPath,
        '-C', 'Release',
        '--output-on-failure'
    )
}

$exe = Get-ChildItem -LiteralPath $buildPath -Recurse -File -Filter 'voxcpm-c.exe' |
    Sort-Object FullName |
    Select-Object -First 1
$library = Get-ChildItem -LiteralPath $buildPath -Recurse -File -Filter 'voxcpm.lib' |
    Sort-Object FullName |
    Select-Object -First 1
if (-not $exe -or -not $library) {
    throw 'Release build did not produce voxcpm-c.exe and voxcpm.lib'
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$packageName = 'voxcpm-c-windows-x64'
$packageRoot = Join-Path $outputPath $packageName
$zipPath = Join-Path $outputPath "$packageName.zip"
foreach ($path in @($packageRoot, $zipPath)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

$includeDir = Join-Path $packageRoot 'include'
New-Item -ItemType Directory -Path $includeDir -Force | Out-Null
Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $packageRoot 'voxcpm-c.exe')
Copy-Item -LiteralPath $library.FullName -Destination (Join-Path $packageRoot 'voxcpm.lib')
Copy-Item -LiteralPath (Join-Path $repoRoot 'include\voxcpm.h') -Destination $includeDir
foreach ($document in @(
    'LICENSE',
    'README.md',
    'c_api.md',
    'THIRD_PARTY_NOTICES.md'
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $document) -Destination $packageRoot
}

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal

$forbidden = @(Get-ChildItem -LiteralPath $packageRoot -Recurse -File | Where-Object {
    $_.Extension -in @('.gguf', '.wav', '.npy', '.bin', '.raw') -or
    $_.Name -match '(fixture|dump)'
})
if ($forbidden.Count -ne 0) {
    throw "Package contains forbidden generated/model files: $($forbidden.FullName -join ', ')"
}

Write-Host "Release package: $zipPath"
Write-Host "Model tests: $(if ($modelPath) { 'enabled' } else { 'not registered (model-free package)' })"
