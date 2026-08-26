<#
.SYNOPSIS
    Register the recorder watcher as a Scheduled Task that starts at logon.

.DESCRIPTION
    Creates a task that launches Watch-Recorder.ps1 hidden at logon. Idempotent: run it
    again to update the task rather than duplicate it.

    Logon trigger plus a resident WMI watcher, rather than a device-arrival event
    trigger: the event-trigger route needs an XML event filter against Kernel-PnP that
    is fiddly to get right and silently matches nothing when wrong. A watcher that
    reports its own startup in a log is easier to trust.

    Runs as the interactive user, NOT SYSTEM. The pipeline needs the user's .env and
    writes into the user's project directory; SYSTEM would have neither.

.PARAMETER Serial
    Volume serial of the recorder.

.PARAMETER Uninstall
    Remove the task.

.EXAMPLE
    # from an elevated OR normal prompt (per-user task needs no elevation)
    .\Install-Watcher.ps1 -Serial AA986EA1

.EXAMPLE
    .\Install-Watcher.ps1 -Uninstall
#>

[CmdletBinding()]
param(
    [string]$Serial = "AA986EA1",
    [string]$ProjectDir = "",
    [string]$Python = "C:\Python313\python.exe",
    [string]$TaskName = "AutoWork Recorder Watcher",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# Default computed here, not in the param block: under Windows PowerShell 5.1 invoked
# via -File, $PSScriptRoot is empty at parameter-binding time and Split-Path throws.
if (-not $ProjectDir) { $ProjectDir = Split-Path -Parent $PSScriptRoot }

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        "removed scheduled task: $TaskName"
    } else {
        "no such scheduled task: $TaskName"
    }
    return
}

$watcher = Join-Path $PSScriptRoot "Watch-Recorder.ps1"
foreach ($p in @($watcher, $Python)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}

$arguments = @(
    "-NoProfile"
    "-NonInteractive"
    "-WindowStyle", "Hidden"
    "-ExecutionPolicy", "Bypass"
    "-File", "`"$watcher`""
    "-Serial", $Serial
    "-ProjectDir", "`"$ProjectDir`""
    "-Python", "`"$Python`""
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)   # 0 = no limit; a long day is a long run

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Watches for the voice recorder (volume $Serial) and runs the AutoWork pipeline. Does not execute action items." `
    -Force | Out-Null

"registered scheduled task: $TaskName"
"  serial      $Serial"
"  project     $ProjectDir"
"  python      $Python"
""
"It starts at your next logon. To start it now without logging out:"
"  Start-ScheduledTask -TaskName '$TaskName'"
""
"To check it is alive:"
"  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
"  Get-Content '$ProjectDir\logs\watcher.log' -Tail 20"
""
"To test the whole path without waiting for a logon:"
"  .\Watch-Recorder.ps1 -Serial $Serial -ProjectDir '$ProjectDir' -Once"
