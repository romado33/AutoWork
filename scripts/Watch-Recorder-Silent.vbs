' Launch Watch-Recorder.ps1 with no console window.
' Task Scheduler + powershell -WindowStyle Hidden still allocates a console on an
' Interactive logon task; Write-Output then keeps that window visible. wscript
' Run style 0 does not.
'
' Args: <serial> <projectDir> <python>
' Exit codes: 1 = missing args, 2 = Watch-Recorder.ps1 not found.
Option Explicit

If WScript.Arguments.Count < 3 Then WScript.Quit 1

Dim serial, projectDir, python, ps1, cmd, sh
serial = WScript.Arguments(0)
projectDir = WScript.Arguments(1)
python = WScript.Arguments(2)
ps1 = projectDir & "\scripts\Watch-Recorder.ps1"

Dim fso
Set fso = CreateObject("Scripting.FileSystemObject")
If Not fso.FileExists(ps1) Then WScript.Quit 2

Set sh = CreateObject("WScript.Shell")
cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File """ _
    & ps1 & """ -Serial " & serial _
    & " -ProjectDir """ & projectDir & """" _
    & " -Python """ & python & """"
' 0 = hidden window; True = wait so the scheduled task stays Running.
sh.Run cmd, 0, True
