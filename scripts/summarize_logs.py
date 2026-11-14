"""Summarise the BEST validation metrics of SLURM logs as a markdown table (mean +- std over seeds).

Each run's log has a header ``[dataset=...] method=... [epoch=...] seed=...`` (written by the
run_*.sh scripts) and one ``[tag] BEST val/acc=..`` line per training phase; the last phase
is reported.  Only finished runs count, and a re-submitted (dataset, method, config, seed)
keeps the newest job.  ``Acc C4`` is the accuracy on the C4-augmented test set, ``Acc orig``
on the original test images (logs without ``acc_c4=`` only have the latter).  ``Time`` is
the SLURM wall-clock (Start -> End time), so only compare runs on the same hardware.

Usage (from the repository root):
    python scripts/summarize_logs.py "logs/rebuttal_histo_*.out" [--verbose]
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np

HEADER_RE = re.compile(
    r"^(?:dataset=(?P<dataset>\S+) )?method=(?P<method>\S+)(?: epoch=(?P<epoch>\S+))? seed=(?P<seed>\S+)"
)
BEST_RE = re.compile(
    r"^\[(?P<tag>\w+)\] BEST val/acc=(?P<acc>[\d.]+)% @ epoch (?P<best_epoch>\d+) \| "
    r"osp=(?P<osp>[\d.]+) sym_kl=(?P<kl>[\d.]+) val_loss=(?P<loss>[\d.]+)"
    r"(?: \| acc_c4=(?P<acc_c4>[\d.]+)% acc_orig=(?P<acc_orig>[\d.]+)%)?"
)
RUN_NAME_RE = re.compile(r"^run_name=(?P<run_name>\S+)")
TIME_RE = re.compile(r"^(?P<which>Start|End) time: (?P<date>.+)$")
JOB_RE = re.compile(r"_(?P<job>\d+)_(?P<task>\d+)\.out$")

# Row order of the table; configurations not listed here are appended alphabetically.
METHOD_ORDER = ["baseline", "avg", "proj", "gcnn", "p4cnn"]


def parse_log(path):
    """Return (key, metrics, job_id) for a finished run, or None."""
    header, best, run_name, times = None, None, None, {}
    with open(path) as f:
        for line in f:
            if header is None and (m := HEADER_RE.match(line)):
                header = m.groupdict()
            elif run_name is None and (m := RUN_NAME_RE.match(line)):
                run_name = m["run_name"]
            elif m := BEST_RE.match(line):
                best = m.groupdict()  # keep the last phase
            elif m := TIME_RE.match(line):
                times[m["which"]] = _parse_date(m["date"])
    if header is None or best is None or "End" not in times:
        return None
    dataset, method = header["dataset"] or "-", header["method"]
    config = header["epoch"]
    if config is None:
        prefix = f"{dataset}_{method}"
        config = run_name[len(prefix) :].lstrip("_") if run_name and run_name.startswith(prefix) else ""
    key = (dataset, method, config or "-")
    minutes = (times["End"] - times["Start"]).total_seconds() / 60 if "Start" in times else float("nan")
    metrics = {
        "seed": header["seed"],
        "tag": best["tag"],
        "best_epoch": int(best["best_epoch"]),
        "acc_c4": float(best["acc_c4"]) if best["acc_c4"] else None,
        "acc_orig": float(best["acc_orig"] or best["acc"]),
        "osp": float(best["osp"]),
        "kl": float(best["kl"]),
        "minutes": minutes,
    }
    job = JOB_RE.search(os.path.basename(path))
    return key, metrics, int(job["job"]) if job else 0


def _parse_date(text):
    # ``date`` output, e.g. "Sat Nov  7 08:23:09 CET 2026"; the time-zone token is dropped
    # (all tasks run in the same zone, only differences are used).
    parts = text.split()
    return datetime.strptime(" ".join(parts[:4] + parts[-1:]), "%a %b %d %H:%M:%S %Y")


def collect(patterns):
    # (dataset, method, config) -> seed -> (job_id, metrics, path)
    runs = defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            parsed = parse_log(path)
            if parsed is None:
                continue
            key, metrics, job_id = parsed
            prev = runs[key].get(metrics["seed"])
            if prev is None or job_id > prev[0]:
                runs[key][metrics["seed"]] = (job_id, metrics, path)
    return runs


def _sort_key(key):
    dataset, method, config = key
    rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (dataset, rank, method, int(config) if config.isdigit() else -1, config)


def _fmt(values, digits, scale=1.0, strip_zero=False):
    values = np.asarray(values) * scale
    mean = f"{values.mean():.{digits}f}"
    if len(values) < 2:
        return mean
    std = f"{values.std(ddof=1):.{digits}f}"
    if strip_zero:  # 0.972 +- .002, as in the paper tables
        std = std.lstrip("0") if std.startswith("0.") else std
    return f"{mean}±{std}"


def _fmt_optional(values, digits):
    return "-" if any(v is None for v in values) else _fmt(values, digits)


def to_markdown(runs, verbose=False):
    lines = [
        "| Dataset | Method | Config | Seeds | Acc C4 (%) | Acc orig (%) | OSP | Sym. KL | Time (min) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(runs, key=_sort_key):
        dataset, method, config = key
        entries = [runs[key][s][1] for s in sorted(runs[key])]
        lines.append(
            f"| {dataset} | {method} | {config} | {len(entries)} "
            f"| {_fmt_optional([e['acc_c4'] for e in entries], 1)} "
            f"| {_fmt([e['acc_orig'] for e in entries], 1)} "
            f"| {_fmt([e['osp'] for e in entries], 3, strip_zero=True)} "
            f"| {_fmt([e['kl'] for e in entries], 4, strip_zero=True)} "
            f"| {_fmt([e['minutes'] for e in entries], 0)} |"
        )
    table = "\n".join(lines)
    if not verbose:
        return table
    detail = [
        "",
        "| Dataset | Method | Config | Seed | Phase | Best epoch | Acc C4 (%) | Acc orig (%) | OSP | Sym. KL "
        "| Time (min) | Log |",
    ]
    detail.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(runs, key=_sort_key):
        for seed in sorted(runs[key]):
            _, e, path = runs[key][seed]
            acc_c4 = "-" if e["acc_c4"] is None else f"{e['acc_c4']:.2f}"
            detail.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {seed} | {e['tag']} | {e['best_epoch']} "
                f"| {acc_c4} | {e['acc_orig']:.2f} | {e['osp']:.4f} | {e['kl']:.4f} | {e['minutes']:.0f} "
                f"| {os.path.basename(path)} |"
            )
    return table + "\n" + "\n".join(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+", help="Glob(s) of SLURM .out logs (quote them)")
    parser.add_argument("--verbose", action="store_true", help="Also print one row per seed with its log file")
    args = parser.parse_args()
    runs = collect(args.patterns)
    if not runs:
        raise SystemExit(f"No finished runs found in {args.patterns}")
    print(to_markdown(runs, verbose=args.verbose))


if __name__ == "__main__":
    main()
