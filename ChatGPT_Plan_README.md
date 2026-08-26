# Work Recorder Automation

A Windows workflow for a USB voice recorder:

**Plug in recorder → copy new audio → Vibe transcribes locally → Ollama summarizes locally → email the daily summary.**

The source recordings are never deleted automatically.

## What you need

1. **Windows 10/11**
2. **Vibe** installed for local transcription
3. **Ollama** installed and running
4. Your USB voice recorder
5. Optional: an SMTP-capable email account if you want automatic email delivery

## Install

1. Extract this ZIP to a folder.
2. Install Vibe and Ollama first.
3. Plug the recorder into the computer.
4. Double-click **INSTALL.cmd**.
5. Follow the prompts.

Setup will:

- copy the automation scripts to `%LOCALAPPDATA%\WorkRecorderAutomation\app`
- create a data folder (default: `Documents\WorkRecorder`)
- download a Whisper transcription model through Vibe
- download your chosen Ollama summary model
- identify the recorder by its volume serial number so the drive letter may change
- optionally configure encrypted SMTP credentials
- install a hidden watcher that starts when you sign in to Windows

## Normal daily use

1. Turn the recorder on when you start recording.
2. Turn it off when finished.
3. Plug it into the Windows computer.
4. Walk away.

The watcher detects the recorder and processes only files it has not handled before.

Default output:

```text
Documents\WorkRecorder\
├── Audio\YYYY-MM-DD\
├── Transcripts\YYYY-MM-DD\
│   └── combined-transcript.md
├── Summaries\
│   └── YYYY-MM-DD.md
├── Models\
├── Logs\
└── state.json
```

If several days of unprocessed recordings are on the recorder, each day is summarized separately.

## Summary behavior

Long days are handled in stages so the local model does not have to ingest an entire workday at once:

1. transcript is split into manageable chunks
2. Ollama extracts tasks, commitments, requests, deadlines, decisions, facts, names, follow-ups and unresolved issues from each chunk
3. extracted notes are deduplicated and merged
4. a final daily work-memory summary is created

The final summary prioritizes:

- top next actions
- things you committed to
- requests made of you
- deadlines/dates
- decisions
- important context
- people to follow up with
- open questions
- possible forgotten follow-ups
- a brief conversation timeline

## Email

Email is optional. Setup includes presets for Gmail, Outlook.com/Hotmail, Microsoft 365 and custom SMTP.

The SMTP password is **not stored as plain text**. PowerShell encrypts it with Windows DPAPI so it can normally only be decrypted by the same Windows user on the same computer.

For Gmail, use a Google **App Password** rather than your normal account password.

By default, only the summary is placed in the email body. Attaching the full transcript is optional and disabled by default.

## iPhone location reminders

The recorder itself has no GPS or phone connection, so it cannot automatically start/stop based on location. A useful companion setup is:

- iPhone Shortcuts automation: **Arrive at work → Show Notification: Turn work recorder ON**
- iPhone Shortcuts automation: **Leave work → Show Notification: Turn work recorder OFF**

You can add a time window to the arrival automation if desired.

## Manual test

With the recorder connected, run:

```text
Run-Now.cmd
```

To test email only, run this from the installed app folder:

```powershell
powershell.exe -ExecutionPolicy Bypass -File Send-TestEmail.ps1
```

## Change settings later

Configuration is stored at:

```text
%LOCALAPPDATA%\WorkRecorderAutomation\app\config.json
```

Useful options:

- `Language`: `en`, or `auto` to detect language
- `EnhanceAudio`: `true` or `false`
- `OllamaModel`: local model name
- `OllamaContext`: context size
- `EmailEnabled`
- `AttachSummaryFile`
- `AttachTranscript`

Run `INSTALL.cmd` again if you want to re-select the recorder, transcription model, Ollama model, or email settings.

## Uninstall watcher

Run **UNINSTALL.cmd**. This removes the startup watcher but intentionally does not delete your audio, transcripts, summaries, models or configuration.

## Privacy / workplace use

Vibe transcription and Ollama summarization remain on your computer. Email delivery necessarily sends the final summary through your email provider. Do not record conversations unless your use is permitted by applicable law, workplace policies, confidentiality obligations and any relevant consent requirements.
