#!/usr/bin/env python3
"""Transcript -> a daily work summary, via a cloud LLM backend.

DIFFERENT FROM EXTRACTION, and the difference drives the design:

  * Extraction answers "what must I do", and only needs the commitment-bearing passages,
    so it runs on prefiltered text (~18% of the transcript).
  * Summarization answers "what happened", and a summary built from 18% of a
    conversation would be confidently incomplete -- worse than no summary, because it
    reads as complete. So this stage sends the WHOLE transcript.

That makes summarization the largest disclosure in the pipeline. It is also why local
inference was never viable for it: at the measured local rate a day's transcript would
take hours, and the prefilter trick that rescued extraction cannot be applied here.

If the summary is going to be emailed, note the disclosure has already happened: the
mail provider sees the same content. The transcript and the audio are what stay local.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from autowork.extract import _normalise as _normalise_for_matching, collapse_same_recording
from autowork.llm import Backend, LLMError, build_backend

logger = logging.getLogger(__name__)


def _action_source_name(action: object) -> str:
    return Path(getattr(action.provenance, "source_audio", "")).name


def _action_origin(action: object) -> str:
    """Human-readable recording stamp for an action, derived from the filename."""
    name = _action_source_name(action)
    stamp = SummaryMeta.from_filename(name).recorded_at
    if stamp is None:
        return name or "unknown recording"
    return f"{stamp.strftime('%A %d %B %Y, %H:%M')} · {name}"


DEFAULT_BACKEND = "openai:gpt-5.4-mini"

# A day's transcript can exceed a comfortable single request. Chunk, summarise each part,
# then summarise the summaries -- the hierarchical approach, which is also what keeps a
# long day from being compressed into uselessly generic bullets.
CHUNK_CHARS = 40_000

SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["label", "summary"],
                "additionalProperties": False,
            },
        },
        # Decisions carry a verbatim supporting quote, checked against the transcript
        # after generation -- the same grounding discipline the action extractor uses.
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["decision", "quote"],
                "additionalProperties": False,
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
        # The honest "nothing here" channel: an accidental or garbled recording gets
        # empty arrays and a one-line reason, not forced content.
        "note": {"type": ["string", "null"]},
    },
    "required": ["headline", "topics", "decisions", "open_questions", "note"],
    "additionalProperties": False,
}

# Operator-authored (2026-08-26), with two amendments: "headline" retained because the
# email subject is built from it, and the JSON shape matches the enforced schema.
# {date} is substituted at run time with the recording date, when known.
SUMMARY_PROMPT = """\
You are an experienced chief of staff summarising a transcript of a spoken work
conversation for the person who recorded it. Capture everything they will need
later: distinct projects and workstreams, systems, customers, numbers, options
considered, what was settled, and what was left open. Compress filler and
repetition, not substance.

Recording date: {date}

Write for someone who was present and wants a complete reminder, not a
headline-only digest. Be specific: name the systems, customers, numbers and
people actually mentioned. "Discussed the mapping accuracy" is useless;
"Entity mapping: ~28% exact match globally, ~90% scoped to one customer's
history" is what they need.

One topic per distinct project, workstream, or decision-thread. When speakers
explicitly introduce distinct projects ("I have two new projects", "the other
project", "that's all the questions I have on that one"), those are separate
topics. Do not merge two named projects into one bullet, and do not fold a new
project into leftover discussion of earlier work. A follow-up call often starts
with leftover work from last week, then moves to new projects: leftover work
and each new project are separate topics. Do not pour a new project's details
into the leftover-work topic because they were discussed in the same sitting.
A finished piece of work from a previous week is its own topic, or omitted,
never mixed into a new project's summary. The headline should name each
distinct project if there are two or three.

Give each topic enough room to keep the useful details: tools named, metrics
cited, options considered and why they were kept or dropped, constraints
(access, licences), and any next step that was spoken. A short paragraph is
better than dropping a fact. Do not merge unrelated threads to stay under a
topic count. Skip only greetings, weather, and garbled stretches.

When speakers walk through a numbered list or feature table, keep a clearly
labelled stretch per row they actually discussed: the feature name, what
counts as a use, the source, the status, and any change they asked for. Two
tables discussed in sequence are two sections, not one. Do not invent
discussion of rows that were not spoken.

Rules:
- Only state what the transcript directly supports. Do not infer or fill gaps;
  omit anything uncertain rather than interpreting it.
- The transcript is from a pocket microphone and contains transcription errors.
  If a passage is garbled, leave it out rather than guessing what it meant.
- Never attribute a statement or commitment to a named person unless the
  transcript makes the speaker unambiguous. Otherwise state it without
  attribution.
- Do not include an action-item list. A separate stage extracts those with
  supporting quotes; generating them twice produces two lists that disagree.
- "decisions" means something actually settled, not merely discussed. An empty
  list is common and correct. Each decision carries the verbatim sentence from
  the transcript where it was settled, copied exactly -- including any
  transcription errors. It is string-matched against the source; a cleaned-up
  quote will fail.
- When a date, deadline or timeframe was spoken ("by Friday", "next week"),
  keep it as spoken, inside the topic or decision it belongs to.
- "open_questions" are questions the conversation left genuinely unresolved.
  Keep each distinct open question; do not collapse two into one.
- Do not produce a list of people mentioned. Names belong inline where they
  came up. Never add titles or turn a company name like "Carter Lumber" into a
  person. Copy names from the transcript, except: when the conversation is
  clearly about a known person or system, use the canonical spelling from the
  known-names list below even if Whisper wrote a near-miss. Do not introduce a
  name that was not discussed.
- If the transcript contains no substantive work content (accidental recording,
  mostly garbled), return empty arrays and set "note" to a one-line reason.
  Fluent topic-salad is garbled: grammatical sentences that jump between unrelated
  subjects (satellites, video games, a backlog item) with no followable thread are
  not a meeting. Do not summarise them. Empty arrays plus a note.

{glossary}

Return only a JSON object with this shape:
{
  "headline": "one specific sentence naming the main outcome",
  "topics": [{"label": "...", "summary": "..."}],
  "decisions": [{"decision": "...", "quote": "..."}],
  "open_questions": ["..."],
  "note": null
}
"""

MERGE_PROMPT = """\
You are merging several partial summaries of ONE conversation into a single summary.

The parts are sequential slices of the same discussion, so the same topic will often \
appear in more than one part. Merge those into one entry rather than repeating them. \
Distinct projects and workstreams stay separate: leftover work from last week \
is not the same topic as a new project introduced later in the same call. Keep \
the specifics -- numbers, names, systems, options considered, constraints. \
When a part is a row-by-row feature-table walkthrough, keep each discussed \
row; do not collapse a table walkthrough into one sentence. Drop nothing \
substantive.

Return JSON in the same shape.
"""


def load_summary_prompt() -> str:
    """The summary prompt, from config/prompts/summary.txt when present.

    Externalised so the operator can paste in any prompt they prefer -- including ones
    published by others -- without touching code. Two things do NOT move with it, on
    purpose, so a pasted prompt cannot weaken them:

      * The output schema is enforced server-side (SUMMARY_SCHEMA). A prompt asking for
        free prose still comes back as the structured sections the email renders.
      * The metadata header (date, duration, speaker count) is derived from the files
        and the diarizer, never from the model, whatever the prompt says.

    One known conflict to avoid when pasting: prompts that ask for an action-item list.
    The review queue owns those, with verbatim grounding quotes; a second list generated
    here would drift from it. The default prompt explains this to the model.
    """
    path = Path(__file__).resolve().parent.parent / "config" / "prompts" / "summary.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return SUMMARY_PROMPT


def fill_summary_prompt(
    template: str,
    *,
    date_text: str,
    glossary=None,
) -> str:
    """Substitute {date} and {glossary}. A pasted prompt without {glossary} still
    gets the known-names block appended, so operator edits cannot drop it."""
    hint = glossary.summary_hint() if glossary is not None else ""
    text = template.replace("{date}", date_text)
    if "{glossary}" in text:
        return text.replace("{glossary}", hint or "Known names: none listed.")
    if hint:
        return f"{text.rstrip()}\n\n{hint}\n"
    return text


class SummaryError(RuntimeError):
    """Summarisation failed. Raised rather than returning an empty summary, because a
    blank summary and a failed request must not look the same in an inbox."""


@dataclass
class SummaryMeta:
    """Factual context for a summary. Derived, never model-generated.

    The split matters. A date, a duration, a filename and a speaker count are FACTS
    available from the recording and the diarizer -- asking a language model for them
    invites a confidently wrong timestamp, and a wrong date on a work summary is worse
    than no date because it gets trusted and filed.

    Calendar invitees, when present, are a separate derived field: they come from
    Outlook for the overlapping meeting, labelled 'from calendar', and are not
    inferred from names spoken in the transcript.
    """

    recorded_at: datetime | None = None
    source_files: list[str] = field(default_factory=list)
    audio_minutes: float = 0.0
    speaker_count: int = 0
    segment_count: int = 0
    meeting_title: str | None = None
    invitees: list[str] = field(default_factory=list)

    @property
    def when(self) -> str:
        if self.recorded_at is None:
            return "unknown (filename not parsed)"
        return self.recorded_at.strftime("%A %d %B %Y, %H:%M")

    @property
    def date_only(self) -> str:
        if self.recorded_at is None:
            return datetime.now().strftime("%Y-%m-%d")
        return self.recorded_at.strftime("%Y-%m-%d")

    @classmethod
    def from_filename(cls, name: str) -> "SummaryMeta":
        """Parse the recorder's R/V<timestamp> naming. None rather than a guess."""
        stem = Path(name).stem
        try:
            stamp = datetime.strptime(stem[1:20], "%Y-%m-%d-%H-%M-%S")
        except (ValueError, IndexError):
            stamp = None
        return cls(recorded_at=stamp, source_files=[Path(name).name])


@dataclass
class Summary:
    headline: str
    topics: list[dict]
    # Each decision: {"decision": str, "quote": str, "verified": bool}. The quote is
    # checked against the source transcript by ground(); "verified" records the result.
    decisions: list[dict]
    open_questions: list[str]
    note: str | None = None
    meta: SummaryMeta = field(default_factory=SummaryMeta)
    # Glossary flywheel, derived from the transcript, never from the model.
    clarify_terms: list[str] = field(default_factory=list)
    backend: str = ""
    elapsed_sec: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "Summary":
        # Decisions arrive as {decision, quote} objects; plain strings (an older
        # prompt, or a pasted custom one without quotes) are accepted and simply
        # cannot be verified.
        decisions: list[dict] = []
        for entry in raw.get("decisions", []):
            if isinstance(entry, dict):
                decisions.append(
                    {
                        "decision": str(entry.get("decision", "")).strip(),
                        "quote": str(entry.get("quote", "")).strip(),
                        "verified": False,
                    }
                )
            elif str(entry).strip():
                decisions.append(
                    {"decision": str(entry).strip(), "quote": "", "verified": False}
                )

        # Topics accept both key styles: the operator-authored prompt uses
        # label/summary, older prompts used topic/detail. Normalised here so a pasted
        # custom prompt cannot break rendering.
        topics = []
        for t in raw.get("topics", []):
            if isinstance(t, dict):
                topics.append({
                    "label": str(t.get("label", t.get("topic", "?"))).strip(),
                    "summary": str(t.get("summary", t.get("detail", ""))).strip(),
                })
        note = raw.get("note")
        return cls(
            headline=str(raw.get("headline", "")).strip(),
            topics=topics,
            decisions=decisions,
            open_questions=[str(q) for q in raw.get("open_questions", [])],
            note=str(note).strip() if note else None,
        )

    def ground(self, transcript: str) -> None:
        """Verify decision and date quotes against the transcript, in place.

        The same discipline the action extractor applies: a model can invent a
        decision, but inventing one whose supporting quote survives a normalised
        substring check against the source is much harder. Unverified entries are
        KEPT and labelled, not dropped -- a summary is read by a human, and "the model
        says this was decided but could not show where" is information.
        """
        haystack = _normalise_for_matching(transcript)
        for entry in self.decisions:
            entry["verified"] = bool(
                entry.get("quote")
                and _normalise_for_matching(entry["quote"]) in haystack
            )

    def apply_glossary(self, glossary) -> None:
        """Rewrite known variants in generated prose. Decision quotes stay verbatim.

        Quotes are string-matched against the transcript; cleaning them here would
        make a fabricated quote look grounded. Headlines and topics are not matched,
        so Cazal -> Casal belongs there once the glossary knows the name.
        """
        self.headline, _ = glossary.correct(self.headline)
        for topic in self.topics:
            topic["label"], _ = glossary.correct(topic.get("label", ""))
            topic["summary"], _ = glossary.correct(topic.get("summary", ""))
        for entry in self.decisions:
            entry["decision"], _ = glossary.correct(entry.get("decision", ""))
        self.open_questions = [
            glossary.correct(q)[0] for q in self.open_questions
        ]
        if self.note:
            self.note, _ = glossary.correct(self.note)

    def partition_actions(self, actions: list | None) -> tuple[list, list]:
        """Split pending items into this recording-set vs earlier days.

        The email used to dump the entire pending queue under today's headline.
        Measured: a 26 August Okta call with Andrew arrived with yesterday's Dan
        mapping/backlog items in the same To do list, and read as one meeting.
        The queue stays the source of truth; earlier items move to a labelled
        section instead of disappearing (a previous crash sent an email with an
        empty to-do list while the queue still held work).

        If this summary has no source_files yet, every action stays in the main
        list -- hiding the queue behind a missing filename would be worse.
        """
        if not actions:
            return [], []
        sources = {Path(name).name.lower() for name in self.meta.source_files if name}
        if not sources:
            return list(actions), []
        today: list = []
        earlier: list = []
        for action in actions:
            if _action_source_name(action).lower() in sources:
                today.append(action)
            else:
                earlier.append(action)
        return collapse_same_recording(today), collapse_same_recording(earlier)

    def to_markdown(self, actions: list | None = None) -> str:
        lines = [f"# {self.headline}", ""]
        lines += [f"- {line}" for line in self.meta_lines()]
        lines.append("")

        today, earlier = self.partition_actions(actions)
        if today:
            lines += [f"## To do ({len(today)})", ""]
            for action in today:
                lines += [
                    f"### {action.title}",
                    "",
                    action.body,
                    "",
                    f"- Target: `{action.target_system}` / {action.action_type.value}"
                    f" / confidence {action.confidence:.2f}",
                    f'- Said: _"{action.provenance.transcript_excerpt}"_',
                    "",
                ]
            lines += ["_Not executed. Review with `scripts\\review-ui.bat` (or `scripts\\review.bat`)._", ""]

        if self.topics:
            lines += ["## Discussed", ""]
            for topic in self.topics:
                lines.append(f"### {topic.get('label', '?')}")
                lines.append("")
                lines.append(topic.get("summary", ""))
                lines.append("")
        if self.decisions:
            lines += ["## Decided", ""]
            for entry in self.decisions:
                flag = "" if entry.get("verified") else " _(unverified: no supporting quote found)_"
                lines.append(f"- {entry.get('decision', '')}{flag}")
            lines.append("")
        if self.open_questions:
            lines += ["## Left open", "", *(f"- {q}" for q in self.open_questions), ""]

        if earlier:
            lines += [
                f"## Still pending from earlier recordings ({len(earlier)})",
                "",
                "Not from this recording. Still awaiting review.",
                "",
            ]
            for action in earlier:
                lines += [
                    f"### {action.title}",
                    "",
                    f"- From: {_action_origin(action)}",
                    "",
                    action.body,
                    "",
                    f"- Target: `{action.target_system}` / {action.action_type.value}"
                    f" / confidence {action.confidence:.2f}",
                    f'- Said: _"{action.provenance.transcript_excerpt}"_',
                    "",
                ]
            if not today:
                lines += ["_Not executed. Review with `scripts\\review-ui.bat` (or `scripts\\review.bat`)._", ""]

        if self.clarify_terms and self._has_discussion():
            lines += [
                "## Terms to clarify",
                "",
                "Capitalised words the glossary does not know. Skim and promote real names.",
                "",
                *(f"- {word}" for word in self.clarify_terms),
                "",
            ]
        return "\n".join(lines)

    def to_html(self, actions: list | None = None) -> str:
        """HTML mail body. The plain-text rendering remains the fallback alternative.

        Exists because plain text in mail clients soft-wraps long lines against the
        hanging indents, producing ragged, hard-to-scan output on phones -- observed on
        a real delivery. Deliberately minimal inline-styled HTML: no CSS classes, no
        images, nothing a strict client will strip.
        """
        import html as _html

        def esc(value: str) -> str:
            return _html.escape(str(value))

        parts = [
            f"<h2 style='margin:0 0 8px'>{esc(self.headline)}</h2>",
            "<table style='color:#555;font-size:13px;border-spacing:0'>",
        ]
        for line in self.meta_lines():
            label, _, value = line.partition("  ")
            parts.append(
                f"<tr><td style='padding-right:12px;white-space:nowrap'><b>{esc(label)}</b></td>"
                f"<td>{esc(value.strip())}</td></tr>"
            )
        parts.append("</table>")
        if self.note:
            parts.append(f"<p style='color:#a60'><b>Note:</b> {esc(self.note)}</p>")

        today, earlier = self.partition_actions(actions)
        if today:
            parts.append(f"<h3>To do ({len(today)})</h3><ol>")
            for action in today:
                parts.append(
                    f"<li style='margin-bottom:10px'><b>{esc(action.title)}</b><br>"
                    f"{esc(action.body)}<br>"
                    f"<span style='color:#777;font-size:12px'>"
                    f"{esc(action.target_system)} / {esc(action.action_type.value)} / "
                    f"confidence {action.confidence:.2f}</span><br>"
                    f"<i style='color:#555'>said: “{esc(action.provenance.transcript_excerpt)}”</i></li>"
                )
            parts.append(
                "</ol><p style='color:#777;font-size:12px'>None of these have been "
                "executed. Review with <code>scripts\\review-ui.bat</code> "
                "(or <code>scripts\\review.bat</code>).</p>"
            )

        if self.topics:
            parts.append("<h3>Discussed</h3>")
            for topic in self.topics:
                parts.append(
                    f"<h4 style='margin:16px 0 6px'>{esc(topic.get('label', '?'))}</h4>"
                    f"<p style='margin:0 0 8px'>{esc(topic.get('summary', ''))}</p>"
                )
        if self.decisions:
            parts.append("<h3>Decided</h3><ul>")
            for entry in self.decisions:
                flag = (
                    "" if entry.get("verified")
                    else " <i style='color:#a00'>(unverified: no supporting quote found)</i>"
                )
                parts.append(f"<li>{esc(entry.get('decision', ''))}{flag}</li>")
            parts.append("</ul>")
        if self.open_questions:
            parts.append("<h3>Left open</h3><ul>")
            parts += [f"<li>{esc(q)}</li>" for q in self.open_questions]
            parts.append("</ul>")

        if earlier:
            parts.append(
                f"<h3>Still pending from earlier recordings ({len(earlier)})</h3>"
                "<p style='color:#777;font-size:12px'>Not from this recording. "
                "Still awaiting review.</p><ol>"
            )
            for action in earlier:
                parts.append(
                    f"<li style='margin-bottom:10px'><b>{esc(action.title)}</b><br>"
                    f"<span style='color:#777;font-size:12px'>"
                    f"From: {esc(_action_origin(action))}</span><br>"
                    f"{esc(action.body)}<br>"
                    f"<span style='color:#777;font-size:12px'>"
                    f"{esc(action.target_system)} / {esc(action.action_type.value)} / "
                    f"confidence {action.confidence:.2f}</span><br>"
                    f"<i style='color:#555'>said: “{esc(action.provenance.transcript_excerpt)}”</i></li>"
                )
            parts.append("</ol>")
            if not today:
                parts.append(
                    "<p style='color:#777;font-size:12px'>None of these have been "
                    "executed. Review with <code>scripts\\review-ui.bat</code> "
                    "(or <code>scripts\\review.bat</code>).</p>"
                )

        if self.clarify_terms and self._has_discussion():
            parts.append(
                "<h3>Terms to clarify</h3>"
                "<p style='color:#777;font-size:12px'>Glossary does not know these yet. "
                "Promote real names into <code>config/glossary.yml</code>.</p><ul>"
            )
            parts += [f"<li>{esc(word)}</li>" for word in self.clarify_terms]
            parts.append("</ul>")
        return "\n".join(parts)

    def meta_lines(self) -> list[str]:
        """Factual header. Every value here is derived, not model-generated.

        Participants and mentions are deliberately separate lines, because a combined
        "Names" list was read as an attendee list on a real two-person call that
        discussed four absent colleagues. The diarizer's voice count is the only
        participant signal a recording actually carries; everything else is who came up
        in conversation, present or not.
        """
        m = self.meta
        lines = [f"When          {m.when}"]
        if m.audio_minutes:
            lines.append(f"Duration      {m.audio_minutes:.0f} min of speech")
        if m.speaker_count:
            # From the diarizer: distinct voices heard on the recording. This is the
            # participant count, as far as audio can know it.
            lines.append(f"Participants  {m.speaker_count} (distinct voices heard)")
        if m.meeting_title:
            lines.append(f"Meeting       {m.meeting_title}")
        if m.invitees:
            # Calendar invite, not names spoken, not diarization labels.
            lines.append(
                f"Invitees      {', '.join(m.invitees)} (from calendar)"
            )
        if m.source_files:
            lines.append(f"Recording     {', '.join(m.source_files)}")
        return lines

    def _has_discussion(self) -> bool:
        """Salad from an unintelligible transcript is not a glossary flywheel.

        The 2026-08-27 11:30 resend listed Nintendo, Starlink and God on an
        empty summary. Those words were Whisper loops, not names to promote.
        """
        return bool(self.topics or self.decisions or self.open_questions)

    def to_text(self, actions: list | None = None) -> str:
        """Plain-text mail body. Must be readable without markdown rendering.

        `actions` are pending items from the review queue. Only those whose source
        recording is in this summary's source_files appear under TO DO. The rest go
        in a labelled earlier-recordings section so one meeting's headline does not
        absorb another day's commitments.
        """
        lines = [f"TOPIC       {self.headline}", *self.meta_lines(), ""]
        if self.note:
            lines += [f"NOTE: {self.note}", ""]

        today, earlier = self.partition_actions(actions)
        if today:
            lines += ["=" * 68, f"TO DO ({len(today)})", "=" * 68, ""]
            for index, action in enumerate(today, start=1):
                lines.append(f"{index}. {action.title}")
                lines.append(f"   {action.body}")
                lines.append(
                    f"   target: {action.target_system} / "
                    f"{action.action_type.value} / confidence {action.confidence:.2f}"
                )
                lines.append(f'   said: "{action.provenance.transcript_excerpt}"')
                lines.append("")
            lines += [
                "None of these have been executed. Review with:",
                "  scripts\\review-ui.bat",
                "  scripts\\review.bat",
                "",
            ]

        if self.topics:
            lines += ["=" * 68, "DISCUSSED", "=" * 68, ""]
            for topic in self.topics:
                lines.append(topic.get("label", "?"))
                lines.append(f"  {topic.get('summary', '')}")
                lines.append("")
        if self.decisions:
            lines.append("DECIDED")
            for entry in self.decisions:
                flag = "" if entry.get("verified") else "  [UNVERIFIED: no supporting quote found]"
                lines.append(f"  * {entry.get('decision', '')}{flag}")
                if entry.get("verified") and entry.get("quote"):
                    lines.append(f'    said: "{entry["quote"]}"')
            lines.append("")
        if self.open_questions:
            lines += ["LEFT OPEN", *(f"  * {q}" for q in self.open_questions), ""]

        if earlier:
            lines += [
                "=" * 68,
                f"STILL PENDING FROM EARLIER RECORDINGS ({len(earlier)})",
                "=" * 68,
                "Not from this recording. Still awaiting review.",
                "",
            ]
            for index, action in enumerate(earlier, start=1):
                lines.append(f"{index}. {action.title}")
                lines.append(f"   from: {_action_origin(action)}")
                lines.append(f"   {action.body}")
                lines.append(
                    f"   target: {action.target_system} / "
                    f"{action.action_type.value} / confidence {action.confidence:.2f}"
                )
                lines.append(f'   said: "{action.provenance.transcript_excerpt}"')
                lines.append("")
            if not today:
                lines += [
                    "None of these have been executed. Review with:",
                    "  scripts\\review-ui.bat",
                    "  scripts\\review.bat",
                    "",
                ]

        if self.clarify_terms and self._has_discussion():
            lines += [
                "TERMS TO CLARIFY",
                "  Skim and promote real names into config/glossary.yml:",
                *(f"  * {word}" for word in self.clarify_terms),
                "",
            ]
        return "\n".join(lines)


def _chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) > size and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _ask(backend: Backend, prompt: str) -> tuple[dict, float, int | None, int | None]:
    import json

    try:
        completion = backend.complete(prompt, schema=SUMMARY_SCHEMA)
    except LLMError as exc:
        raise SummaryError(f"summarisation failed: {exc}") from exc
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        raise SummaryError(f"summary was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError(f"expected a JSON object, got {type(data).__name__}")
    return data, completion.elapsed_sec, completion.input_tokens, completion.output_tokens


def summarise(
    text: str,
    backend_spec: str = DEFAULT_BACKEND,
    recorded_at: datetime | None = None,
    glossary=None,
    require_intelligible: bool = True,
    extra_instructions: str = "",
) -> Summary:
    """Summarise a transcript, chunking and merging if it is long."""
    if not text.strip():
        raise SummaryError("nothing to summarise: the transcript is empty")

    from autowork.quality import unintelligible_reason

    # Backstop: the pipeline should have dropped these segments already. If a
    # caller passes leftover salad (or a cached contrib from before the check),
    # do not spend a summary request inventing a meeting.
    # require_intelligible=False is the operator --force-keep path only.
    if require_intelligible:
        reason = unintelligible_reason(text)
        if reason:
            return Summary(
                headline="Recording was too garbled to summarise",
                topics=[],
                decisions=[],
                open_questions=[],
                note=reason,
            )

    backend = build_backend(backend_spec)
    # {date} gives the model the recording date as CONTEXT; the prompt still requires
    # relative dates be kept as spoken, so this cannot become invented absolutes.
    date_text = recorded_at.strftime("%A %d %B %Y") if recorded_at else "unknown"
    prompt_text = fill_summary_prompt(
        load_summary_prompt(), date_text=date_text, glossary=glossary,
    )
    if extra_instructions.strip():
        prompt_text = f"{prompt_text.rstrip()}\n\n{extra_instructions.strip()}\n"
    chunks = _chunk(text)
    logger.info("summarising %d chars in %d chunk(s)", len(text), len(chunks))

    partials: list[dict] = []
    elapsed = 0.0
    tokens_in = tokens_out = 0

    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            f"{prompt_text}\nTRANSCRIPT PART {index} OF {len(chunks)}:\n{chunk}\n\nJSON:"
        )
        data, took, tin, tout = _ask(backend, prompt)
        partials.append(data)
        elapsed += took
        tokens_in += tin or 0
        tokens_out += tout or 0

    if len(partials) == 1:
        merged = partials[0]
    else:
        import json

        prompt = (
            f"{MERGE_PROMPT}\nPARTIAL SUMMARIES:\n"
            f"{json.dumps(partials, indent=1)}\n\nJSON:"
        )
        merged, took, tin, tout = _ask(backend, prompt)
        elapsed += took
        tokens_in += tin or 0
        tokens_out += tout or 0

    summary = Summary.from_dict(merged)
    summary.ground(text)
    summary.backend = f"{backend.name}:{backend.model}"
    summary.elapsed_sec = elapsed
    summary.input_tokens = tokens_in or None
    summary.output_tokens = tokens_out or None

    if not summary.headline and not summary.note:
        raise SummaryError("the model returned a summary with no headline and no note")
    if not summary.headline:
        summary.headline = f"No substantive work content ({summary.note})"
    return summary
