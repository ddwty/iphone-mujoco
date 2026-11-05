"""Source-release boundaries: preserve private workbench files without shipping them."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from check_repo_hygiene import check_repository, is_public_path
from export_source import export_source


def _write(root, name, value="local data"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def test_clean_export_keeps_only_authored_source_and_preserves_local_data(tmp_path):
    _write(tmp_path, "pyproject.toml", '[project]\nname="iphone-mujoco"\nversion="0.3.0"\n')
    _write(tmp_path, "README.md", "# Public source\n")
    _write(tmp_path, "LICENSE", "MIT\n")
    _write(tmp_path, "src/iphone_mujoco/__init__.py", '__version__="0.3.0"\n')
    _write(tmp_path, "src/iphone_mujoco/assets/iphone.xml", "<mujocoinclude/>\n")
    _write(tmp_path, "Open Phone Lab.command", "#!/bin/bash\nexit 0\n").chmod(0o755)
    private = ["profiles/device.json", "recordings/private/frame.json", "validation/report.json",
               ".venv/lib/device.py", ".cache/model.xml", "scenes/generated.xml",
               "src/iphone_mujoco.egg-info/PKG-INFO", "src/iphone_mujoco/__pycache__/cache.pyc"]
    for name in private:
        _write(tmp_path, name)
    archive, manifest_path = export_source(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.namelist()
        assert len(entries) == 6
        assert all(not any(part in entry.split("/") for part in ("profiles", "recordings", "validation", ".cache", ".venv")) for entry in entries)
        assert bundle.getinfo("iphone-mujoco-0.3.0/Open Phone Lab.command").external_attr >> 16 & 0o111
    manifest = json.loads(manifest_path.read_text())
    assert manifest["file_count"] == 6 and len(manifest["archive_sha256"]) == 64
    assert all((tmp_path / name).read_text() == "local data" for name in private)


def test_source_findings_report_location_without_sensitive_value(tmp_path):
    address = "192.168." + "10.29"
    user_path = "/" + "Users/" + "private-developer/project"
    secret = "ghp_" + "A" * 35
    _write(tmp_path, "examples/config.py", f'address="{address}"\npath="{user_path}"\nsecret="{secret}"\n')
    findings, _ = check_repository(tmp_path, mode="source")
    assert {finding.line for finding in findings} == {1, 2, 3}
    displayed = "\n".join(finding.display() for finding in findings)
    assert address not in displayed and user_path not in displayed and secret not in displayed
    assert all("examples/config.py:" in finding.display() for finding in findings)
    assert not is_public_path("profiles/phone.json")
    assert not is_public_path("src/copied_sdk/phone_pb2.py")


def test_staged_check_reads_index_and_rejects_force_added_recording(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = _write(tmp_path, "README.md", "No private settings.\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    address = "192.168." + "10.29"
    path.write_text(f'address="{address}"\n')
    working_tree_findings, _ = check_repository(tmp_path, mode="tracked")
    assert any("private device IP" in finding.reason for finding in working_tree_findings)
    findings, _ = check_repository(tmp_path, mode="staged")
    assert not findings  # Only the clean indexed content is being checked.
    _write(tmp_path, "recordings/frame.json")
    subprocess.run(["git", "-C", str(tmp_path), "add", "recordings/frame.json"], check=True)
    findings, _ = check_repository(tmp_path, mode="staged")
    assert any(finding.path == "recordings/frame.json" for finding in findings)
