"""Distribution checks must count bundled libraries and treat each wheel separately."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("check_wheel_size", SCRIPTS / "check_wheel_size.py")
size = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size)


def wheel(tmp_path, target="x86_64", library_bytes=1000):
    path = tmp_path / f"tensorcodec-1.0-cp310-abi3-manylinux2014_{target}.whl"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("tensorcodec/", b"")
        archive.writestr("tensorcodec/__init__.py", b"# codec\n")
        archive.writestr("tensorcodec.libs/libavcodec.so.62", b"x" * library_bytes)
    return path


def test_counts_bundled_libraries_without_extraction(tmp_path):
    path = wheel(tmp_path)
    metrics = size.measure_wheel(path)
    assert metrics["download_bytes"] == path.stat().st_size
    assert metrics["unpacked_bytes"] == 1008
    assert not (tmp_path / "tensorcodec.libs").exists()


def test_limit_is_inclusive_and_checks_both_dimensions(tmp_path):
    metrics = size.measure_wheel(wheel(tmp_path))
    policy = {"max_" + field: metrics[field] for field in ("download_bytes", "unpacked_bytes")}
    assert size.check_sizes(metrics, policy) == []
    policy["max_download_bytes"] -= 1
    policy["max_unpacked_bytes"] -= 1
    assert len(size.check_sizes(metrics, policy)) == 2


def test_cli_checks_each_architecture_and_fails_on_one_oversize_wheel(tmp_path):
    paths = [wheel(tmp_path), wheel(tmp_path, "aarch64", library_bytes=2000)]
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_download_bytes": 10000, "max_unpacked_bytes": 1500}))
    output = tmp_path / "reports/size.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "check_wheel_size.py"),
            *map(str, paths),
            "--policy",
            str(policy),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert len(report["wheels"]) == 2
    assert len(report["failures"]) == 1
    assert "aarch64" in report["failures"][0]


def test_cli_does_not_sum_separate_wheels(tmp_path):
    paths = [wheel(tmp_path), wheel(tmp_path, "aarch64")]
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_download_bytes": 10000, "max_unpacked_bytes": 1500}))
    subprocess.run(
        [sys.executable, str(SCRIPTS / "check_wheel_size.py"), *map(str, paths), "--policy", str(policy)],
        check=True,
        capture_output=True,
    )


def test_invalid_archive_fails(tmp_path):
    path = tmp_path / "broken.whl"
    path.write_bytes(b"not a ZIP archive")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_wheel_size.py"), str(path)], capture_output=True, check=False
    )
    assert result.returncode != 0


def test_reports_delta_by_architecture(tmp_path):
    metric = size.measure_wheel(wheel(tmp_path, "aarch64"))
    baseline = {"tensorcodec": {"version": "0.1", "wheels": [{**metric, "download_bytes": 0}]}}
    report = size.format_report([metric], baseline)
    assert "aarch64" in report and "published TensorCodec 0.1" in report
    assert "| +0.00 | +0.00 |" in report


@pytest.mark.parametrize("filename", ["codec.whl", "codec-win_amd64.whl"])
def test_unsupported_architecture_is_explicit(filename):
    with pytest.raises(ValueError, match="architecture"):
        size.architecture(filename)
