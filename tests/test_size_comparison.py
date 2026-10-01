"""Verify published-wheel selection and guarded documentation updates without networking."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# The CLI imports its sibling script; make that module available without changing sys.path.
for name in ("check_wheel_size", "update_size_comparison"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
comparison = sys.modules["update_size_comparison"]


def artifact(target, python="cp310", platform="manylinux2014", yanked=False):
    return {
        "filename": f"codec-1.0-{python}-abi3-{platform}_{target}.whl",
        "packagetype": "bdist_wheel",
        "yanked": yanked,
    }


def test_selects_same_python_and_linux_architectures():
    wanted = [artifact("x86_64"), artifact("aarch64")]
    others = [
        artifact("x86_64", python="cp311"),
        artifact("x86_64", platform="macosx"),
        artifact("aarch64", yanked=True),
    ]
    assert comparison.select_wheels({"urls": wanted + others}, "cp310") == wanted


@pytest.mark.parametrize(
    "items", [[], [artifact("x86_64")], [artifact("x86_64"), artifact("x86_64"), artifact("aarch64")]]
)
def test_missing_or_ambiguous_wheels_fail(items):
    with pytest.raises(ValueError, match="Expected one"):
        comparison.select_wheels({"urls": items}, "cp310")


def test_replaces_only_marked_content():
    assert comparison.replace_block("before<start>old<end>after", "<start>", "<end>", "new") == (
        "before<start>\nnew\n<end>after"
    )


@pytest.mark.parametrize("text", ["missing", "<start><start><end>", "<end><start>"])
def test_invalid_markers_fail(text):
    with pytest.raises(ValueError, match="ordered"):
        comparison.replace_block(text, "<start>", "<end>", "new")


def test_reference_version_matches_playback_oracle():
    import json

    root = SCRIPTS.parent
    version = json.loads((root / "packaging/size-policy.json").read_text())["comparison"]["version"]
    assert f'"torchcodec=={version}"' in (root / "pyproject.toml").read_text()


@pytest.mark.parametrize("failure", ["hash", "size", None])
def test_measures_published_wheel_and_verifies_hash_and_size(tmp_path, monkeypatch, failure):
    import hashlib
    from zipfile import ZipFile

    items = []
    for target in comparison.TARGETS:
        item = artifact(target)
        path = tmp_path / item["filename"]
        with ZipFile(path, "w") as archive:
            archive.writestr("codec.libs/library.so", b"codec")
        item.update(
            {
                "url": path.as_uri(),
                "size": path.stat().st_size,
                "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            }
        )
        items.append(item)
    if failure == "hash":
        items[0]["digests"]["sha256"] = "invalid"
    elif failure == "size":
        items[0]["size"] += 1
    monkeypatch.setattr(comparison, "release_metadata", lambda *_: {"urls": items})
    if failure:
        with pytest.raises(ValueError, match="hash/size mismatch"):
            comparison.measure_release("codec", "1.0", "cp310")
    else:
        release = comparison.measure_release("codec", "1.0", "cp310")
        assert len(release["wheels"]) == 2
        assert all(wheel["unpacked_bytes"] == 5 for wheel in release["wheels"])
        assert release["wheels"][0]["sha256"] == items[0]["digests"]["sha256"]
