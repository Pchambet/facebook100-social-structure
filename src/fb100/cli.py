"""Command line entry point: `uv run fb100 {data,run,figures,report}`."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "raw" / "facebook100"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"
SITE_DIR = ROOT / "site"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fb100", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="download and extract Facebook100 (idempotent)")
    run = sub.add_parser("run", help="run every analysis and write results/")
    run.add_argument("--jobs", type=int, default=None, help="worker processes (default 3)")
    run.add_argument("--limit", type=int, default=None, help="only the N smallest campuses")
    sub.add_parser("figures", help="draw the static figures in docs/figures/")
    sub.add_parser("report", help="build the static report site/index.html")
    args = parser.parse_args(argv)

    if args.command == "data":
        from fb100.data import fetch

        fetch(DATA_DIR)
    elif args.command == "run":
        from fb100.pipeline import default_jobs
        from fb100.pipeline import run as run_pipeline

        run_pipeline(DATA_DIR, RESULTS_DIR, jobs=args.jobs or default_jobs(), limit=args.limit)
    elif args.command == "figures":
        from fb100.figures import draw_all

        draw_all(RESULTS_DIR, FIGURES_DIR)
    elif args.command == "report":
        from fb100.report import build

        build(RESULTS_DIR, SITE_DIR / "index.html")


if __name__ == "__main__":
    main()
