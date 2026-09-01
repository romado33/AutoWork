# AutoWork

Pocket voice recorder to reviewed action items, by way of a transcript and a daily
summary email.

```
recorder --> ingest --> AUDIO GATE --> slice+compress --> transcribe (cloud)
                            |                                    |
                     (drops silence &                     glossary correction
                      handling noise)                             |
                                                        RELEVANCE GATE
                                                      (drops podcasts, ambient)
                                                               |
                                            +------------------+------------------+
                                            |                                     |
                                        summarise                         prefilter + extract
                                            |                                     |
                                            +------------> EMAIL <-------- review queue
                                                                          (nothing executed)
```

## Quick start

```bat
scripts\setup.bat            :: once: venv, dependencies, self-test
notepad .env                 :: add OPENAI_API_KEY and your Gmail app password
scripts\run.bat              :: plug in the recorder, then run
scripts\review-ui.bat        :: browser: review and mark to-dos done
scripts\review.bat           :: same thing in the terminal
```

To make it automatic on every login:

```powershell
.\scripts\Install-Watcher.ps1 -Serial AA986EA1
```

## Prerequisites

| Need | Why | Notes |
|---|---|---|
| Python 3.11+ | everything | tick "Add to PATH" when installing |
| ffmpeg | the gate and the slicing/compression | on PATH, or drop `ffmpeg.exe` in this folder |
| OpenAI API key | transcription, summary, extraction | platform.openai.com/api-keys |
| Gmail App Password | the summary email | myaccount.google.com/apppasswords, needs 2FA |

Three pip packages: `openai`, `PyYAML`, `pytest`. No local models, no GPU, no torch.

## The two gates, and why there are two

They answer different questions and both were added because of a real failure.

**The audio gate** (`autowork/gate.py`) asks *is there speech-shaped signal here*, from
the spectrum, for free, before anything is uploaded. Two conditions, both required:

- `speech_db >= -32` — is there speech energy at all?
- `speech_db - rumble_db >= +1` — is that energy speech rather than rumble?

Either alone gives false positives in opposite directions, and both were observed.
Calibrated against samples with known transcription outcomes; see `tests/test_gate.py`,
where every row is a real measurement whose transcript was read.

This gate is also what controls cost, because transcription is billed per minute of
audio. On a real 45-minute recording it dropped 49% as silence and handling noise.

**The relevance gate** (`autowork/relevance.py`) asks *is this a work conversation*,
after transcription, from the words. The audio gate cannot answer this, because the
answer is not in the audio. Measured on real recordings:

| Input | Verdict |
|---|---|
| Work conversation, rambling and full of filler | keep (0.99) |
| An hour of true-crime podcast, flawless audio | drop, media_playback (0.99) |
| Ambient noise transcribed as "Beep beep. There it is man." | drop, ambient_noise (0.96) |

## Why the transcript can lie, and what stops it

Whisper on low-signal audio does not return empty. It returns fluent, grammatical,
confident English that nobody said. Measured examples from this project's own
recordings: `*Police*` x6, `city of Estrella` x14, and
`"So, around Monday, we're going to do a lot of things"` — which reads exactly like a
real work commitment.

Four defences, in order of strength:

1. **The audio gate** never sends that audio to be transcribed in the first place.
2. **The relevance gate** drops what does get through and is not a conversation.
3. **Grounding quotes.** Every extracted action must carry a verbatim quote, which is
   then checked against the source transcript. A model can invent an action; inventing
   one whose quote survives a substring check is much harder.
4. **The review queue.** Nothing is executed. Ever. By anything here.

## Costs

Transcription dominates and is billed per minute, so the audio gate is the lever.
Compression cuts upload size, not cost. Measured on a 45-minute recording:

| | Raw | Gated + compressed |
|---|---|---|
| Minutes billed | 45 | 23 |
| Bytes uploaded | 43 MB | ~5.5 MB |

Compression settings were measured, not assumed: 16 kHz mono MP3 at 32 kbps transcribed
within noise of the as-recorded 32 kHz stereo 128 kbps baseline. Mono is free because the
device's two channels are bit-identical (-91 dB apart). 16 kHz is free because Whisper
resamples there internally. Opus at 20 kbps genuinely degraded and is not used.

## What is automated and what is not

Ingest, gating, transcription, summarising, extraction and the emails all run unattended.

USB plug-in sends **one email per conversation**. A conversation is one recording,
or several recordings that overlap the **same Outlook meeting** (same start + title).
Similar-sounding topics are not merged -- that is how a mapping call and an Okta
call were once mailed as one meeting.

Outstanding actions that are still pending or approved (not yet done) go out in a
**separate weekday 7:30am digest**. Open `scripts\review-ui.bat` and click
**Mark done** -- no executor runs. The terminal equivalent is
`scripts\review.bat --done <id>`.

If Outlook is installed and signed in, an overlapping calendar event supplies the
meeting title and invitees as derived metadata, labelled "from calendar". Zero or
two-plus overlapping events: those fields are omitted rather than guessed. Diarization
labels A/B are never mapped onto invitees.

**Action items are never executed.** They land in the review queue as `pending`, and
`dispatch()` refuses anything that is not `approved`. Plugging in a USB stick must not be
able to write to Jira.

## Layout

```
autowork/
  gate.py             audio quality gate, calibrated
  transcribe_cloud.py gate -> slice -> compress -> OpenAI
  relevance.py        is this a work conversation?
  glossary.py         two-tier domain glossary, self-improving
  prefilter.py        commitment-language selection (cuts extraction volume ~4x)
  extract.py          transcript -> grounded ActionRecords
  summarize.py        transcript -> conversation summary
  conversations.py    group recordings only when they share an Outlook event
  digest.py           morning outstanding-queue email (no model)
  day.py              per-recording sidecar; rebuilds a date, not a plug-in
  calendar_lookup.py  Outlook invitees as derived metadata (optional)
  action.py           the ActionRecord contract (vendor-neutral)
  queue.py            SQLite review queue, enforced state machine
  executors/          the execution boundary; only APPROVED dispatches
  llm.py              swappable backends (OpenAI, Anthropic)
  mailer.py           Gmail SMTP
  ingest.py           copy from the recorder by volume serial
tools/                CLIs for each stage, plus the localhost review UI
scripts/              .bat entry points, USB watcher, task installer
config/glossary.yml   domain terms; add names as they come up
```

## The glossary

Two tiers, because a decode-time prompt is capped at ~224 tokens:

- `tier: prompt` biases transcription. Fixes words the model never had a chance to get
  right. Measured: "Clark" to "Claude", "personal intelligence tool" to "Customer
  Intelligence Tool".
- `tier: correct` is applied to finished text. Unbounded. Fixes near-misses like
  "forums" to "forms".

Each cloud transcript ends with a **Terms to clarify** list: capitalised words the
glossary does not know. The same list, filtered to repeated terms, is on the email.
Skim it and promote the real names. That is the loop that makes accuracy compound
week over week.

## Portability

No vendor is load-bearing. `llm.py` takes a backend string (`openai:gpt-5.4-mini`,
`anthropic:claude-sonnet-5`), and `executors/` puts a hard interface between the queue
and anything that acts on it. Switching provider is one config value; losing access to a
target system means deleting one adapter, not touching the core.

Local inference was tried and removed. On this hardware gemma3:4b took 358s on a
2,600-character excerpt and found 1 of 3 real action items, qwen2.5:7b took 493s and
found none, and phi4 never finished. The same transcript through gpt-5.4-mini takes ~7s
and finds all three.

## Tests

```bat
.venv\Scripts\python.exe -m pytest tests -q
```

275 tests, no network required. The valuable ones encode real measurements: the gate
calibration table, the verbatim hallucination strings the loop detector must catch, and
the credential mistakes that actually happened (a pasted app password with spaces in it).
