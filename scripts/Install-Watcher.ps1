<#
.SYNOPSIS
    Register the recorder watcher as a Scheduled Task that starts at logon.

.DESCRIPTION
    Creates a task that launches Watch-Recorder.ps1 hidden at logon. Idempotent: run it
    again to update the task rather than duplicate it.

    Launched via Watch-Recorder-Silent.vbs (wscript Run style 0), not powershell
    -WindowStyle Hidden: an Interactive scheduled task still allocates a console, and
    the watcher's heartbeat Write-Output then keeps that window visible.

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
    [string]$DigestTaskName = "AutoWork Morning Queue",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# Default computed here, not in the param block: under Windows PowerShell 5.1 invoked
# via -File, $PSScriptRoot is empty at parameter-binding time and Split-Path throws.
if (-not $ProjectDir) { $ProjectDir = Split-Path -Parent $PSScriptRoot }

if ($Uninstall) {
    foreach ($name in @($TaskName, $DigestTaskName)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            "removed scheduled task: $name"
        } else {
            "no such scheduled task: $name"
        }
    }
    return
}

$watcher = Join-Path $PSScriptRoot "Watch-Recorder.ps1"
$silent = Join-Path $PSScriptRoot "Watch-Recorder-Silent.vbs"
foreach ($p in @($watcher, $silent, $Python)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}

# wscript.exe //B: no script UI. The .vbs hides the powershell console.
$arguments = @(
    "//B"
    "//nologo"
    "`"$silent`""
    $Serial
    "`"$ProjectDir`""
    "`"$Python`""
) -join " "

$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument $arguments
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

$digestScript = Join-Path $ProjectDir "tools\send_queue_digest.py"
if (-not (Test-Path $digestScript)) { throw "not found: $digestScript" }
$digestAction = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-u `"$digestScript`"" `
    -WorkingDirectory $ProjectDir
$digestTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At 7:30am
Register-ScheduledTask `
    -TaskName $DigestTaskName `
    -Action $digestAction `
    -Trigger $digestTrigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Weekday 7:30am email of outstanding AutoWork actions. Does not execute them." `
    -Force | Out-Null

"registered scheduled task: $DigestTaskName"
"  weekdays    07:30 local, outstanding PENDING + APPROVED items"
""
"It starts at your next logon. To start the watcher now without logging out:"
"  Start-ScheduledTask -TaskName '$TaskName'"
""
"To send the morning queue now:"
"  Start-ScheduledTask -TaskName '$DigestTaskName'"
"  or:  $Python -u `"$digestScript`""
""
"To check the watcher is alive:"
"  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
"  Get-Content '$ProjectDir\logs\watcher.log' -Tail 20"
""
"To test the whole path without waiting for a logon:"
"  .\Watch-Recorder.ps1 -Serial $Serial -ProjectDir '$ProjectDir' -Once"
