<#
.SYNOPSIS
    Watches for the voice recorder being plugged in and runs the pipeline.

.DESCRIPTION
    Registers a WMI event subscription on Win32_VolumeChangeEvent and fires the pipeline
    when a volume whose serial matches -Serial arrives.

    Matched by VOLUME SERIAL, not drive letter. The recorder mounts as whatever letter is
    free, so a hardcoded D:\ works until something else takes D: -- at which point it
    would happily ingest a different USB stick.

    A short settle delay before acting: Windows raises the arrival event before the
    filesystem is reliably readable, and enumerating too early sees an empty volume.

.PARAMETER Serial
    Volume serial of the recorder. Find it with:
        Get-CimInstance Win32_LogicalDisk | Select DeviceID,VolumeSerialNumber

.PARAMETER ProjectDir
    AutoWork directory containing tools\run_pipeline.py.

.PARAMETER Once
    Handle one arrival and exit. Useful for testing without a resident process.

.EXAMPLE
    .\Watch-Recorder.ps1 -Serial AA986EA1 -ProjectDir C:\Users\RobDods\Apps\ClaudeCode\AutoWork

.NOTES
    Runs the pipeline, which ingests, transcribes, summarises, extracts and emails.
    It does NOT execute action items -- those land in the review queue. Plugging in a
    USB stick must not be able to write to Jira.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Serial,
    [Parameter(Mandatory = $true)][string]$ProjectDir,
    [string]$Python = "C:\Python313\python.exe",
    [int]$SettleSeconds = 8,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$Serial = $Serial.ToUpper().Replace("-", "")
$LogDir = Join-Path $ProjectDir "logs"
$null = New-Item -ItemType Directory -Force -Path $LogDir

function Write-Log {
    param([string]$Message)
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    # Add-Content with explicit UTF8: Tee-Object in Windows PowerShell 5.1 writes
    # UTF-16, which reads as spaced-out garbage in every other tool.
    Add-Content -Path (Join-Path $LogDir "watcher.log") -Value $line -Encoding UTF8
    Write-Output $line
}

function Test-Prerequisites {
    if (-not (Test-Path $Python)) { throw "Python not found at $Python" }
    $script = Join-Path $ProjectDir "tools\run_pipeline.py"
    if (-not (Test-Path $script)) { throw "pipeline not found at $script" }
    if (-not (Test-Path (Join-Path $ProjectDir ".env"))) {
        Write-Log "WARNING: no .env in $ProjectDir; the pipeline will fail without OPENAI_API_KEY"
    }
    # Fail now rather than at 6pm when a recording is waiting.
    & $Python -c "import openai, yaml" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "python deps missing: run scripts\setup.bat" }
}

function Get-RecorderDrive {
    Get-CimInstance Win32_LogicalDisk |
        Where-Object { $_.VolumeSerialNumber -and
                       $_.VolumeSerialNumber.ToUpper().Replace("-", "") -eq $Serial } |
        Select-Object -First 1 -ExpandProperty DeviceID
}

function Invoke-Pipeline {
    $drive = Get-RecorderDrive
    if (-not $drive) {
        Write-Log "volume $Serial not present after settle; nothing to do"
        return
    }
    Write-Log "recorder found at $drive - running pipeline"
    Push-Location $ProjectDir
    try {
        # Output is captured to a dated log: a watcher that fails silently is worse
        # than no watcher, and there is no console attached when run as a task.
        $log = Join-Path $LogDir ("run-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
        # Launched via cmd /c with file redirection, NOT a PowerShell pipeline. In
        # Windows PowerShell 5.1, redirecting a native command's stderr (*>&1) wraps
        # each line in an ErrorRecord, and under $ErrorActionPreference = "Stop" the
        # first INFO log line -- Python logging writes to stderr -- killed this watcher
        # mid-run twice. Measured, not hypothetical: ingest completed, transcription
        # never started, task exited 1 with no run log at all.
        cmd /c "`"$Python`" -u tools\run_pipeline.py --serial $Serial -v > `"$log`" 2>&1"
        Write-Log "pipeline exited $LASTEXITCODE (log: $log)"
    }
    finally { Pop-Location }
}

Test-Prerequisites
Write-Log "watcher started (serial $Serial, project $ProjectDir)"

# If the recorder is already attached at startup, handle it: the arrival event fired
# before this process existed and will not be replayed.
if (Get-RecorderDrive) {
    Write-Log "recorder already attached at startup"
    # try/catch here as well as in the event loop: a throw on the startup path killed
    # the watcher with no log line saying why. A dead watcher must always say so.
    try { Invoke-Pipeline }
    catch { Write-Log "pipeline error at startup: $($_.Exception.Message)" }
    if ($Once) { return }
}

# EventType 2 is "volume arrived". Polling the WMI query every 5s is what
# __InstanceCreationEvent WITHIN requires; the event itself is push-based from there.
$query = "SELECT * FROM Win32_VolumeChangeEvent WHERE EventType = 2"
Register-WmiEvent -Query $query -SourceIdentifier "RecorderArrival" -SupportEvent

try {
    while ($true) {
        $null = Wait-Event -SourceIdentifier "RecorderArrival"
        Remove-Event -SourceIdentifier "RecorderArrival"

        Write-Log "volume arrival detected; waiting ${SettleSeconds}s to settle"
        Start-Sleep -Seconds $SettleSeconds

        try { Invoke-Pipeline }
        catch { Write-Log "pipeline error: $($_.Exception.Message)" }

        if ($Once) { break }
    }
}
finally {
    Unregister-Event -SourceIdentifier "RecorderArrival" -ErrorAction SilentlyContinue
    Write-Log "watcher stopped"
}
