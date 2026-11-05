"""Export reviewed public source with per-file hashes and a source manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

from check_repo_hygiene import check_repository


# Archive permissions are part of the release, not the exporting host's modes.
# Windows does not retain POSIX execute bits, and source ZIPs need no Git index.
EXECUTABLE_SOURCE = frozenset({"Open Phone Lab.command", "run.command", "setup.command"})


def project_version(root: Path) -> str:
    content = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', content)
    if not match or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)", match.group(1)):
        raise ValueError("Expected an explicit project version in pyproject.toml")
    return match.group(1)


def export_source(root: Path, output: Path | None = None, *, overwrite: bool = False) -> tuple[Path, Path]:
    root = Path(root).resolve()
    version = project_version(root)
    output = Path(output) if output else root / "dist" / f"iphone-mujoco-source-{version}.zip"
    if output.suffix.lower() != ".zip":
        raise ValueError("Source export output must end in .zip")
    manifest_path = output.with_suffix(".manifest.json")
    checksum_path = output.with_suffix(".sha256")
    if not overwrite and any(path.exists() for path in (output, manifest_path, checksum_path)):
        raise FileExistsError("Source archive/manifest already exists; choose a new path or pass --overwrite")
    findings, paths = check_repository(root, mode="source")
    selected = {path.relative_to(root).as_posix() for path in paths}
    for required in ("LICENSE", "README.md", "src/iphone_mujoco/__init__.py"):
        if required not in selected:
            raise ValueError("Required public source file is missing: " + required)
    if findings:
        detail = "\n".join(finding.display() for finding in findings)
        raise ValueError("Public source hygiene failed; no archive written:\n" + detail)
    files = []
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"iphone-mujoco-{version}"
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                relative = path.relative_to(root).as_posix()
                payload = path.read_bytes()
                mode = 0o100755 if relative in EXECUTABLE_SOURCE else 0o100644
                info = zipfile.ZipInfo(f"{prefix}/{relative}", date_time=(2000, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = mode << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
                files.append({"path": relative, "bytes": len(payload), "mode": oct(mode),
                              "sha256": hashlib.sha256(payload).hexdigest()})
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    archive_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"project": "iphone-mujoco", "version": version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive": output.name, "archive_sha256": archive_hash, "file_count": len(files),
                "selection": "explicit reviewed source allowlist; no workbench data or dependency caches",
                "excluded_roots": ["validation", "recordings", "profiles", "scenes", ".venv", ".cache", "dist", "build"],
                "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
    return output, manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        archive, manifest = export_source(args.root, args.output, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    print(archive)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
