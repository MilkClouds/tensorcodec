"""Measure final wheels without importing packages or extracting their contents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def measure_wheel(path: Path) -> dict:
    with ZipFile(path) as archive:
        unpacked = sum(item.file_size for item in archive.infolist() if not item.is_dir())
    return {"filename": path.name, "download_bytes": path.stat().st_size, "unpacked_bytes": unpacked}


def architecture(filename: str) -> str:
    for target in ("x86_64", "aarch64"):
        if filename.endswith(f"_{target}.whl"):
            return target
    raise ValueError(f"Unsupported wheel architecture: {filename}")


def check_sizes(metrics: dict, policy: dict) -> list[str]:
    return [
        f"{metrics['filename']}: {field} = {metrics[field]} bytes exceeds {policy['max_' + field]} bytes"
        for field in ("download_bytes", "unpacked_bytes")
        if metrics[field] > policy["max_" + field]
    ]


def format_report(wheels: list[dict], baseline: dict) -> str:
    previous = {architecture(w["filename"]): w for w in baseline.get("tensorcodec", {}).get("wheels", [])}
    lines = [
        "## TensorCodec wheel size",
        "",
        "Final repaired wheels; dependencies inside the archive are included. NumPy is excluded.",
        "",
        "| Architecture | Download (MiB) | Unpacked (MiB) | Download delta (MiB) | Unpacked delta (MiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for wheel in wheels:
        target = architecture(wheel["filename"])
        old = previous.get(target)
        sizes = [f"{wheel[key] / 2**20:.2f}" for key in ("download_bytes", "unpacked_bytes")]
        deltas = [
            f"{(wheel[key] - old[key]) / 2**20:+.2f}" if old else "—" for key in ("download_bytes", "unpacked_bytes")
        ]
        lines.append(f"| {target} | {' | '.join(sizes + deltas)} |")
    if previous:
        lines += ["", f"Delta baseline: published TensorCodec {baseline['tensorcodec']['version']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--policy", type=Path, default=ROOT / "packaging/size-policy.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "packaging/size-baseline.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    wheels = [measure_wheel(path) for path in args.wheels]
    failures = [message for wheel in wheels for message in check_sizes(wheel, policy)]
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    report = format_report(wheels, baseline)
    report += f"\nLimits per wheel: {policy['max_download_bytes'] / 2**20:g} MiB download, "
    report += f"{policy['max_unpacked_bytes'] / 2**20:g} MiB unpacked.\n"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"wheels": wheels, "failures": failures}, indent=2) + "\n")
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
