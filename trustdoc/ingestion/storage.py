"""Source naming rules and safe on-disk storage of original uploads.

* ``Source.name`` is the *only* identifier used in citations: a normalised,
  project-relative POSIX path. Absolute paths and ``..`` are rejected.
* Originals are stored content-addressed under
  ``<uploads>/project_<id>/<sha256><ext>``. The on-disk filename is derived
  from the hash and a fixed extension, never from user input, so uploads cannot
  escape the uploads directory or overwrite each other. ``Source.stored_path``
  holds the path *relative to the uploads directory*.
"""

from __future__ import annotations

import os
import shutil
import re
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath

from trustdoc.exceptions import CleanupError, InvalidSourceNameError, UnsupportedSourceError
from trustdoc.models import SourceType

MAX_NAME_LENGTH = 512
MAX_SEGMENT_LENGTH = 255

_EXTENSIONS: dict[str, SourceType] = {
    ".py": SourceType.PYTHON,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
}
_STORED_EXTENSION = {
    SourceType.PYTHON: ".py",
    SourceType.MARKDOWN: ".md",
    SourceType.PDF: ".pdf",
}
SUPPORTED_EXTENSIONS = tuple(_EXTENSIONS)
_DRIVE = re.compile(r"^[A-Za-z]:")


def sanitize_source_name(name: str) -> str:
    """Return a safe project-relative POSIX name, or raise InvalidSourceNameError."""
    if not isinstance(name, str) or not name.strip():
        raise InvalidSourceNameError("Source name must not be empty")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise InvalidSourceNameError("Source name contains control characters")
    text = unicodedata.normalize("NFC", name.strip()).replace("\\", "/")
    if text.startswith(("/", "~")) or _DRIVE.match(text):
        raise InvalidSourceNameError(f"Source name must be project-relative, got {name!r}")
    segments = [seg for seg in text.split("/") if seg not in ("", ".")]
    if not segments:
        raise InvalidSourceNameError(f"Source name has no file component: {name!r}")
    if ".." in segments:
        raise InvalidSourceNameError(f"Path traversal is not allowed in source names: {name!r}")
    if any(len(seg) > MAX_SEGMENT_LENGTH for seg in segments):
        raise InvalidSourceNameError("A source name component is too long")
    cleaned = "/".join(segments)
    if len(cleaned) > MAX_NAME_LENGTH:
        raise InvalidSourceNameError("Source name is too long")
    return cleaned


def detect_source_type(name: str) -> SourceType:
    ext = PurePosixPath(name).suffix.lower()
    try:
        return _EXTENSIONS[ext]
    except KeyError:
        raise UnsupportedSourceError(
            f"Unsupported file type {ext or '(no extension)'!r} for {name!r}; "
            f"supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        ) from None


class UploadStore:
    """Content-addressed storage for original uploaded files."""

    def __init__(self, uploads_dir: Path) -> None:
        self.root = Path(uploads_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def relative_path(self, project_id: int, sha256: str, source_type: SourceType) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return f"project_{int(project_id)}/{sha256}{_STORED_EXTENSION[source_type]}"

    def save(
        self, project_id: int, sha256: str, source_type: SourceType, data: bytes
    ) -> tuple[str, bool]:
        """Atomically store ``data``. Returns (relative_path, newly_created)."""
        rel = self.relative_path(project_id, sha256, source_type)
        target = self.resolve(rel)
        if target.is_file() and target.stat().st_size == len(data):
            return rel, False
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".upload-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return rel, True

    def resolve(self, stored_path: str) -> Path:
        """Map a stored relative path to a file inside the uploads directory."""
        candidate = (self.root / stored_path).resolve()
        if os.path.isabs(stored_path) or not candidate.is_relative_to(self.root):
            raise InvalidSourceNameError(f"Stored path escapes the uploads directory: {stored_path!r}")
        return candidate

    def read(self, stored_path: str) -> bytes:
        return self.resolve(stored_path).read_bytes()

    def project_dir(self, project_id: int) -> Path:
        """Resolve the safe on-disk directory for a project's uploads."""
        target = (self.root / f"project_{int(project_id)}").resolve()
        if not target.is_relative_to(self.root):
            raise InvalidSourceNameError(f"Invalid project id escapes uploads directory: {project_id}")
        return target

    def delete_project(self, project_id: int) -> bool:
        """Remove the project upload directory and all contained files."""
        proj_dir = self.project_dir(project_id)
        if proj_dir.is_dir():
            try:
                shutil.rmtree(proj_dir)
                return True
            except OSError as exc:
                raise CleanupError(f"Failed to delete upload files for project {project_id}: {exc}") from exc
        return False

    def delete(self, stored_path: str) -> bool:
        path = self.resolve(stored_path)
        if path.is_file():
            try:
                path.unlink()
                return True
            except OSError as exc:
                raise CleanupError(f"Failed to delete stored file {stored_path}: {exc}") from exc
        return False
