"""Refresh the README from hash-verified, published PyPI wheels (standard library only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from check_wheel_size import ROOT, architecture, check_sizes, measure_wheel

START = "<!-- wheel-size:start -->"
END = "<!-- wheel-size:end -->"
BADGE_START = "<!-- wheel-size-badge:start -->"
BADGE_END = "<!-- wheel-size-badge:end -->"
TARGETS = ("x86_64", "aarch64")


def release_metadata(project: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == 9:
                raise
            time.sleep(5)
    raise AssertionError("unreachable")


def select_wheels(metadata: dict, python_tag: str) -> list[dict]:
    selected = []
    for target in TARGETS:
        matches = [
            item
            for item in metadata["urls"]
            if item["packagetype"] == "bdist_wheel"
            and not item.get("yanked", False)
            and f"-{python_tag}-" in item["filename"]
            and "manylinux" in item["filename"]
            and item["filename"].endswith(f"_{target}.whl")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {python_tag} Linux {target} wheel, found {len(matches)}")
        selected.append(matches[0])
    return selected


def measure_release(project: str, version: str, python_tag: str) -> dict:
    metadata = release_metadata(project, version)
    result = {"version": version, "source": f"https://pypi.org/project/{project}/{version}/", "wheels": []}
    with tempfile.TemporaryDirectory() as directory:
        for item in select_wheels(metadata, python_tag):
            path = Path(directory) / item["filename"]
            urllib.request.urlretrieve(item["url"], path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["digests"]["sha256"] or path.stat().st_size != item["size"]:
                raise ValueError(f"PyPI hash/size mismatch: {item['filename']}")
            result["wheels"].append({**measure_wheel(path), "sha256": digest, "url": item["url"]})
    return result


def replace_block(text: str, start: str, end: str, content: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError(f"README must contain one ordered {start} / {end} pair")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + content + "\n" + end + after


def comparison_table(snapshot: dict) -> str:
    lines = [
        (
            f"Published PyPI Linux wheels, CPython 3.10: TensorCodec {snapshot['tensorcodec']['version']} / "
            f"TorchCodec {snapshot['torchcodec']['version']}."
        ),
        "",
        "| Package | Architecture | Download | Unpacked |",
        "| --- | --- | ---: | ---: |",
    ]
    for project, label in (("tensorcodec", "TensorCodec"), ("torchcodec", "TorchCodec")):
        for wheel in snapshot[project]["wheels"]:
            lines.append(
                f"| {label} | {architecture(wheel['filename'])} | "
                f"{wheel['download_bytes'] / 1e6:.1f} MB | {wheel['unpacked_bytes'] / 1e6:.1f} MB |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Already published TensorCodec version")
    parser.add_argument("--policy", type=Path, default=ROOT / "packaging/size-policy.json")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--output", type=Path, default=ROOT / "packaging/size-baseline.json")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    reference = policy["comparison"]
    snapshot = {
        "tensorcodec": measure_release("tensorcodec", args.version, "cp310"),
        "torchcodec": measure_release(reference["project"], reference["version"], reference["python_tag"]),
    }
    failures = [message for wheel in snapshot["tensorcodec"]["wheels"] for message in check_sizes(wheel, policy)]
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")
    largest = max(wheel["download_bytes"] for wheel in snapshot["tensorcodec"]["wheels"])
    badge = (
        f'<a href="#package-size"><img src="https://img.shields.io/badge/wheel-{largest / 1e6:.1f}%20MB-blue" '
        'alt="Wheel download"></a>'
    )
    readme = replace_block(args.readme.read_text(), START, END, comparison_table(snapshot))
    readme = replace_block(readme, BADGE_START, BADGE_END, badge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n")
    args.readme.write_text(readme)
    print(comparison_table(snapshot))


if __name__ == "__main__":
    main()
