' start_arvis.vbs
' -----------------------------------------------------------
' Visible Windows launcher for the arvis assistant.
'
' Double-click this file (or have core.autostart point the
' HKCU Run registry value at it) to start arvis in a
' **visible** console window with ``cd`` echoed, so the user
' can see exactly which folder python is running from.
' -----------------------------------------------------------
Option Explicit

Dim shell, fso, root, bat
Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
bat  = root & "\start_arvis_terminal.bat"

If Not fso.FileExists(bat) Then
    MsgBox "Could not find start_arvis_terminal.bat in " & root, vbCritical, "arvis"
    WScript.Quit 1
End If

' 1 = activate the window, 0 = wait for the batch to exit.
shell.Run Chr(34) & bat & Chr(34), 1, False
