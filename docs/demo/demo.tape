# README animation. Render with: vhs docs/demo/demo.tape  (from the repo root)
# A scripted re-enactment (docs/demo/replay.py), not a live recording.

Output docs/demo/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 720
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set WindowBar Colorful

Hide
Type "export PS1='$ '; lk() { python3 docs/demo/replay.py; }; clear"
Enter
Show

Sleep 500ms
Type "lk agent console"
Sleep 400ms
Enter
Sleep 30s
