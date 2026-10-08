"""Minimal command-line entry point: ``python -m trustdoc <command>``."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from trustdoc import __version__
from trustdoc.app import bootstrap
from trustdoc.config import Settings
from trustdoc.exceptions import TrustDocError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trustdoc", description="TrustDoc AI foundation CLI")
    parser.add_argument("--version", action="version", version=f"trustdoc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create data directories and the SQLite database")
    sub.add_parser("info", help="show resolved settings, schema version and row counts")
    args = parser.parse_args(argv)

    try:
        ctx = bootstrap(Settings.from_env())
    except TrustDocError as exc:
        print(f"error: {exc}")
        return 2

    if args.command == "init":
        print(f"Initialised database at {ctx.settings.storage.db_path} "
              f"(schema v{ctx.db.schema_version()})")
    elif args.command == "info":
        print(json.dumps(
            {
                "version": __version__,
                "schema_version": ctx.db.schema_version(),
                "settings": ctx.settings.to_dict(),
                "table_counts": ctx.db.table_counts(),
            },
            indent=2,
        ))
    return 0
