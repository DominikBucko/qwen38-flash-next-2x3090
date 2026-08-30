#!/usr/bin/env python3
"""Fail until publication-specific cross-links and immutable pins are filled."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    lock = json.loads((ROOT / "repro.lock.json").read_text())
    published = lock["published_model"]
    errors: list[str] = []
    if not published.get("repo"):
        errors.append("published_model.repo is not set in repro.lock.json")
    revision = published.get("revision") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("published_model.revision must be a full 40-character Hub commit")

    placeholders = (
        "OWNER/",
        "YOUR_NAME/",
        "YOUR_HF_NAME/",
        "FULL_40_CHARACTER",
        "FULL_HUGGING_FACE_COMMIT",
        "FULL_COMMIT_OR_TAG",
    )
    for relative in ("README.md", "CITATION.cff", "packaging/model-card.md"):
        text = (ROOT / relative).read_text()
        found = [placeholder for placeholder in placeholders if placeholder in text]
        if found:
            errors.append(f"publication placeholders in {relative}: {found}")

    if errors:
        print("release is not ready:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("release pins and cross-links are complete")


if __name__ == "__main__":
    main()
