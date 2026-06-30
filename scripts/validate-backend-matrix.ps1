[CmdletBinding()]
param(
    [string]$CpuExe = '',
    [string]$CudaExe = '',
    [string]$F16Model = '',
    [string]$Q8Model = '',
    [string]$OutputReport = 'docs/backend-matrix-latest.json',
    [string]$Text = 'VoxCPM backend validation.',
    [int]$Steps = 2,
    [int]$MaxLen = 6,
    [switch]$AllowMissing,
    [string]$ValidateReport = ''
)

$ErrorActionPreference = 'Stop'
$requiredFields = @(
    'name', 'backend', 'model_format', 'status', 'exit_code',
    'wall_ms', 'peak_working_set_mb', 'audio_sec', 'rtf',
    'sample_rate', 'n_samples', 'finite_samples', 'peak', 'rms',
    'clipped_samples'
)

function Test-BackendReport {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        $rows = @(Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    } catch {
        [Console]::Error.WriteLine(
            "invalid JSON report: $($_.Exception.Message)")
        return $false
    }
    if ($rows.Count -eq 0) {
        [Console]::Error.WriteLine(
            'backend report must contain at least one row')
        return $false
    }
    foreach ($row in $rows) {
        foreach ($field in $requiredFields) {
            if ($null -eq $row.PSObject.Properties[$field]) {
                [Console]::Error.WriteLine(
                    "backend report row is missing '$field'")
                return $false
            }
        }
        if ($row.status -notin @('passed', 'failed', 'unavailable')) {
            [Console]::Error.WriteLine(
                "invalid backend report status '$($row.status)'")
            return $false
        }
        if ($row.status -eq 'passed') {
            if ([int]$row.exit_code -ne 0 -or
                [int64]$row.n_samples -le 0 -or
                [int64]$row.finite_samples -ne [int64]$row.n_samples -or
                [double]$row.peak_working_set_mb -le 0.0 -or
                [double]$row.audio_sec -le 0.0 -or
                [double]$row.rtf -le 0.0 -or
                [double]$row.rms -le 0.0) {
                [Console]::Error.WriteLine(
                    "passed row '$($row.name)' has invalid metrics")
                return $false
            }
        }
    }
    return $true
}

if ($ValidateReport) {
    if (Test-BackendReport -Path $ValidateReport) {
        Write-Host "valid backend report: $ValidateReport"
        exit 0
    }
    exit 1
}

if ($Steps -le 0 -or $MaxLen -le 0) {
    throw 'Steps and MaxLen must be positive'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
function Resolve-InputPath {
    param([string]$Path)
    if (-not $Path) {
        return ''
    }
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Read-PcmWavMetrics {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [IO.File]::OpenRead($Path)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        if ([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'RIFF') {
            throw 'missing RIFF header'
        }
        [void]$reader.ReadUInt32()
        if ([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'WAVE') {
            throw 'missing WAVE header'
        }

        $format = 0
        $channels = 0
        $sampleRate = 0
        $bits = 0
        [byte[]]$audioBytes = @()
        while ($stream.Position + 8 -le $stream.Length) {
            $chunkId = [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $chunkSize = $reader.ReadUInt32()
            $chunkStart = $stream.Position
            if ($chunkId -eq 'fmt ') {
                $format = $reader.ReadUInt16()
                $channels = $reader.ReadUInt16()
                $sampleRate = $reader.ReadUInt32()
                [void]$reader.ReadUInt32()
                [void]$reader.ReadUInt16()
                $bits = $reader.ReadUInt16()
            } elseif ($chunkId -eq 'data') {
                $audioBytes = $reader.ReadBytes([int]$chunkSize)
            }
            $stream.Position = $chunkStart + $chunkSize + ($chunkSize % 2)
        }
        if ($format -ne 1 -or $bits -ne 16 -or
            $channels -ne 1 -or $sampleRate -le 0 -or
            $audioBytes.Count -eq 0) {
            throw "expected mono PCM16 WAV, got format=$format channels=$channels bits=$bits"
        }

        $sampleCount = [int64]($audioBytes.Count / 2)
        $sumSq = 0.0
        $peak = 0.0
        $clipped = 0L
        for ($offset = 0; $offset -lt $audioBytes.Count; $offset += 2) {
            $sample = [BitConverter]::ToInt16($audioBytes, $offset)
            $value = [double]$sample / 32768.0
            $absolute = [Math]::Abs($value)
            if ($absolute -gt $peak) {
                $peak = $absolute
            }
            if ([Math]::Abs([int]$sample) -ge 32767) {
                $clipped++
            }
            $sumSq += $value * $value
        }
        return [ordered]@{
            sample_rate = [int]$sampleRate
            n_samples = $sampleCount
            finite_samples = $sampleCount
            peak = $peak
            rms = [Math]::Sqrt($sumSq / [double]$sampleCount)
            clipped_samples = $clipped
            audio_sec = [double]$sampleCount / [double]$sampleRate
        }
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Invoke-MatrixCase {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Case,
        [Parameter(Mandatory = $true)][string]$WorkDir
    )

    $base = [ordered]@{
        name = $Case.name
        backend = $Case.backend
        model_format = $Case.format
        status = 'unavailable'
        exit_code = -1
        wall_ms = 0.0
        peak_working_set_mb = 0.0
        audio_sec = 0.0
        rtf = 0.0
        sample_rate = 0
        n_samples = 0
        finite_samples = 0
        peak = 0.0
        rms = 0.0
        clipped_samples = 0
    }

    if (-not (Test-Path -LiteralPath $Case.exe -PathType Leaf) -or
        -not (Test-Path -LiteralPath $Case.model -PathType Leaf)) {
        if (-not $AllowMissing) {
            $base.status = 'failed'
        }
        return [pscustomobject]$base
    }

    $textPath = Join-Path $WorkDir "$($Case.name)-text.txt"
    $wavPath = Join-Path $WorkDir "$($Case.name).wav"
    $stdoutPath = Join-Path $WorkDir "$($Case.name)-stdout.log"
    $stderrPath = Join-Path $WorkDir "$($Case.name)-stderr.log"
    [IO.File]::WriteAllText(
        $textPath, $Text, [Text.UTF8Encoding]::new($false))

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Case.exe
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in @(
        'tts', '--model', $Case.model, '--text-file', $textPath,
        '--out', $wavPath, '--steps', "$Steps", '--max-len', "$MaxLen",
        '--backend', $Case.backend, '--pcm16', '--no-denoiser'
    )) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $timer = [Diagnostics.Stopwatch]::StartNew()
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $peakWorkingSet = 0L
    while (-not $process.WaitForExit(100)) {
        try {
            $process.Refresh()
            if ($process.WorkingSet64 -gt $peakWorkingSet) {
                $peakWorkingSet = $process.WorkingSet64
            }
        } catch {
            # The process may exit between WaitForExit and Refresh.
        }
    }
    $process.WaitForExit()
    $timer.Stop()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($stdoutPath, $stdout)
    [IO.File]::WriteAllText($stderrPath, $stderr)

    $base.exit_code = $process.ExitCode
    $base.wall_ms = $timer.Elapsed.TotalMilliseconds
    try {
        $reportedPeak = [int64]$process.PeakWorkingSet64
        if ($reportedPeak -gt $peakWorkingSet) {
            $peakWorkingSet = $reportedPeak
        }
    } catch {
        # Keep the sampled peak when the exited process no longer exposes it.
    }
    $base.peak_working_set_mb = [double]$peakWorkingSet / 1MB

    if ($process.ExitCode -eq 0 -and
        (Test-Path -LiteralPath $wavPath -PathType Leaf) -and
        $stderr -notmatch '(GGML_ASSERT|unsupported.+cast|CUDA error|failed to initialize)') {
        try {
            $metrics = Read-PcmWavMetrics -Path $wavPath
            foreach ($key in $metrics.Keys) {
                $base[$key] = $metrics[$key]
            }
            $base.rtf =
                ($base.wall_ms / 1000.0) / $base.audio_sec
            $base.status =
                if ($base.rms -gt 0.000001 -and
                    $base.finite_samples -eq $base.n_samples) {
                    'passed'
                } else {
                    'failed'
                }
        } catch {
            $base.status = 'failed'
        }
    } else {
        $base.status = 'failed'
    }
    return [pscustomobject]$base
}

$cpuExePath = Resolve-InputPath $CpuExe
$cudaExePath = Resolve-InputPath $CudaExe
$f16ModelPath = Resolve-InputPath $F16Model
$q8ModelPath = Resolve-InputPath $Q8Model
$outputPath = Resolve-InputPath $OutputReport
$outputDirectory = Split-Path -Parent $outputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Path $outputDirectory -Force |
        Out-Null
}

$workDir = Join-Path ([IO.Path]::GetTempPath()) "voxcpm-matrix-$PID"
New-Item -ItemType Directory -Path $workDir -Force | Out-Null
try {
    $cases = @(
        @{ name = 'cpu-f16'; backend = 'cpu'; format = 'f16'; exe = $cpuExePath; model = $f16ModelPath },
        @{ name = 'cpu-q8'; backend = 'cpu'; format = 'q8_0'; exe = $cpuExePath; model = $q8ModelPath },
        @{ name = 'cuda-f16'; backend = 'cuda'; format = 'f16'; exe = $cudaExePath; model = $f16ModelPath },
        @{ name = 'cuda-q8'; backend = 'cuda'; format = 'q8_0'; exe = $cudaExePath; model = $q8ModelPath }
    )
    $results = @(
        foreach ($case in $cases) {
            Invoke-MatrixCase -Case $case -WorkDir $workDir
        }
    )
    $results | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $outputPath -Encoding utf8
    if (-not (Test-BackendReport -Path $outputPath)) {
        exit 1
    }
    $results | Format-Table name, status, wall_ms, rtf, rms,
        peak_working_set_mb -AutoSize
    Write-Host "backend report: $outputPath"
    if (@($results | Where-Object status -eq 'failed').Count -gt 0) {
        exit 1
    }
} finally {
    Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue
}
