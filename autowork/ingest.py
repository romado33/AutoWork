#!/usr/bin/env python3
"""Copy new recordings off the device. Identified by volume serial, never by drive letter.

WHY BY SERIAL: the recorder mounts as whatever letter is free, so a hardcoded D:\\ works
until the day something else takes D:. Worse, it would then happily ingest a completely
different USB stick. The volume serial is stable for the life of the format.

WHY THIS EXISTS AT ALL: this step was missing for the first two days of the project, and
its absence lost a recording -- the audio was read straight off the device, analysed, and
then the device was unplugged before anything was copied. The transcript survived; the
audio did not.

NEVER DELETES FROM THE DEVICE. The recorder holds ~2,200 hours; there is no space
pressure worth risking data for, and a copy bug that also deletes is unrecoverable.
Files already ingested are skipped by name plus size, so re-running is free and safe.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# The recorder writes V<timestamp>.MP3 with voice-activation on, R<timestamp>.MP3 with
# continuous recording. Both are recordings; anything else on the volume is not.
AUDIO_SUFFIXES = {".mp3", ".wav"}


class IngestError(RuntimeError):
    """Ingest failed. Raised rather than returning an empty list, so "no new files"
    and "the device was not found" cannot be confused."""


@dataclass
class IngestResult:
    copied: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    source_root: Path | None = None

    @property
    def copied_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.copied if p.exists())


def find_device(volume_serial: str) -> Path | None:
    """Locate the mount point of the volume with this serial. None if not attached.

    Uses PowerShell rather than parsing drive letters, because the serial is the only
    identifier that distinguishes this recorder from any other removable volume.
    """
    wanted = volume_serial.strip().upper().replace("-", "")
    script = (
        "Get-CimInstance Win32_LogicalDisk | "
        "Select-Object DeviceID,VolumeSerialNumber | "
        "ForEach-Object { \"$($_.DeviceID)|$($_.VolumeSerialNumber)\" }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IngestError(f"cannot enumerate volumes: {exc}") from exc

    if proc.returncode != 0:
        raise IngestError(f"volume enumeration failed: {proc.stderr.strip()[:200]}")

    for line in proc.stdout.splitlines():
        if "|" not in line:
            continue
        device, _, serial = line.partition("|")
        if serial.strip().upper().replace("-", "") == wanted:
            return Path(device.strip() + "/")
    return None


def find_recordings(root: Path, subdir: str = "RECORD") -> list[Path]:
    """Recordings on the device. Looks in RECORD/ first, then the volume root.

    Some firmware writes to a RECORD folder, some to the root; checking both means the
    same code works if the device is reformatted or the firmware changes.
    """
    candidates: list[Path] = []
    for base in (root / subdir, root):
        if not base.is_dir():
            continue
        try:
            for entry in base.iterdir():
                if entry.is_file() and entry.suffix.lower() in AUDIO_SUFFIXES:
                    candidates.append(entry)
        except OSError as exc:
            logger.warning("cannot list %s: %s", base, exc)
        if candidates:
            break
    return sorted(candidates)


LEDGER_NAME = ".ingested.json"


def _load_ledger(destination: Path) -> dict[str, int]:
    """Names and sizes of everything ever ingested, surviving local deletion.

    The device never deletes and the operator manages local retention by hand, so
    "already present in dest_dir" is not a durable memory: delete a local copy and the
    next run would resurrect it from the device and re-process it at real cost. The
    ledger remembers what was ingested independently of whether the copy still exists.
    """
    path = destination / LEDGER_NAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        # A corrupt ledger must not brick ingest; worst case is re-copying files that
        # are mostly still present locally and skipped by the name+size check anyway.
        logger.warning("ledger %s unreadable (%s); starting a fresh one", path, exc)
        return {}


def _save_ledger(destination: Path, ledger: dict[str, int]) -> None:
    path = destination / LEDGER_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def ingest(
    volume_serial: str,
    dest_dir: str | os.PathLike,
    min_bytes: int = 100_000,
) -> IngestResult:
    """Copy recordings never ingested before. Returns what was copied.

    "New" means absent from the ledger AND absent from dest_dir: the ledger remembers
    across manual deletions, the dest check self-heals if the ledger is lost. A file is
    only entered in the ledger after its copy has fully succeeded.

    `min_bytes` drops the stub files the recorder leaves when a session is started and
    stopped immediately -- one such 11 KB file was present on the real device. They
    contain no speech and would only waste a gate pass.
    """
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(destination)

    device = find_device(volume_serial)
    if device is None:
        raise IngestError(
            f"no volume with serial {volume_serial!r} is attached. "
            f"Is the recorder plugged in?"
        )

    result = IngestResult(source_root=device)
    for source in find_recordings(device):
        size = source.stat().st_size
        if size < min_bytes:
            logger.info("skipping stub file %s (%d bytes)", source.name, size)
            continue

        target = destination / source.name

        # Ingested before, even if the local copy has since been deleted by hand.
        # Same size means the same recording; a different size under the same name is
        # a new recording and falls through to be copied.
        if ledger.get(source.name) == size:
            result.skipped.append(target)
            continue

        # Name plus size, not name alone: a same-named file of a different size means a
        # different recording, and skipping it would silently lose data.
        if target.exists() and target.stat().st_size == size:
            # Present but not in the ledger (pre-ledger ingest, or a lost ledger):
            # adopt it so a later manual deletion stays deleted.
            ledger[source.name] = size
            result.skipped.append(target)
            continue

        # Copy to a temp name and rename, so an interrupted copy cannot leave a
        # truncated file that the next run would treat as already ingested.
        staging = target.with_suffix(target.suffix + ".part")
        try:
            shutil.copy2(source, staging)
            os.replace(staging, target)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise IngestError(f"failed copying {source.name}: {exc}") from exc

        logger.info("ingested %s (%.1f MB)", source.name, size / 1e6)
        result.copied.append(target)
        # Only after the copy fully succeeded: a ledger entry for a failed copy would
        # permanently hide that recording.
        ledger[source.name] = size

    _save_ledger(destination, ledger)
    logger.info(
        "ingest complete: %d copied, %d already ingested",
        len(result.copied), len(result.skipped),
    )
    return result
