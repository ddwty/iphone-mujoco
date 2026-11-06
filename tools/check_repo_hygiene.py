"""Check public source/index without printing sensitive source values.

No files are deleted or modified. `--mode source` selects an explicit public
allowlist in a dirty workbench; tracked/staged modes also reject indexed files
outside that selection. This is a packaging boundary, not a secret-scanning
service or a substitute for reviewing what will be published.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


ROOT_FILES = frozenset({
    "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "MANIFEST.in", ".gitignore", ".gitattributes", ".editorconfig",
    "setup.command", "run.command", "Open Phone Lab.command",
})
DIRECTORY_SUFFIXES = {
    "src": {".py", ".typed", ".xml", ".json"},
    "tests": {".py", ".xml", ".json"},
    "examples": {".py", ".xml", ".json"},
    "tools": {".py"},
    "docs": {".md", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"},
    ".github": {".md", ".yml", ".yaml"},
}
PRIVATE_ROOTS = frozenset({"validation", "recordings", "profiles", "scenes", "dist", "build"})
IGNORED_PARTS = frozenset({".git", ".venv", "venv", "env", ".cache", "__pycache__",
                           ".pytest_cache", ".ruff_cache", ".mypy_cache", ".idea", ".vscode"})
BINARY_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
HOME_PATH = re.compile(r"/(?:Users|home)/[^/\s\"'<>]+|[A-Za-z]:\\Users\\[^\\\s\"'<>]+")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                    r"pypi-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|"
                    r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----)")
# Network-prefix integers keep the checker's own policy distinct from a device
# address embedded in product code or documentation.
PRIVATE_NETWORKS = tuple(ipaddress.ip_network((prefix, bits)) for prefix, bits in
                         ((0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0xA9FE0000, 16)))


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    line: int | None = None

    def display(self) -> str:
        location = self.path + (f":{self.line}" if self.line else "")
        return f"{location}: {self.reason}"


def is_public_path(relative: str | PurePosixPath) -> bool:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return False
    if any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in path.parts):
        return False
    if path.name in {".DS_Store", "MUJOCO_LOG.TXT", "VALIDATION.md"} or path.name.startswith("._"):
        return False
    if len(path.parts) == 1:
        return path.name in ROOT_FILES
    if path.parts[0] in PRIVATE_ROOTS:
        return False
    if path.parts[0] == "src":
        if len(path.parts) < 3:
            return False
        if path.parts[1] == "asmagic_mujoco":
            return len(path.parts) == 3 and path.name in {"__init__.py", "__main__.py"}
        if path.parts[1] != "iphone_mujoco":
            return False
    return path.suffix.lower() in DIRECTORY_SUFFIXES.get(path.parts[0], set())


def public_source_files(root: Path) -> list[Path]:
    """Select authored public files; never walk local datasets or caches."""
    root = Path(root).resolve()
    result = [root / name for name in ROOT_FILES if (root / name).is_file() or (root / name).is_symlink()]
    for directory in DIRECTORY_SUFFIXES:
        start = root / directory
        if not start.exists():
            continue
        if start.is_symlink():
            result.append(start)
            continue
        for parent, directories, names in os.walk(start, followlinks=False):
            directories[:] = sorted(name for name in directories
                                     if name not in IGNORED_PARTS and not name.endswith(".egg-info"))
            for name in names:
                path = Path(parent) / name
                if is_public_path(path.relative_to(root).as_posix()):
                    result.append(path)
            for name in list(directories):
                path = Path(parent) / name
                if path.is_symlink():
                    result.append(path)
                    directories.remove(name)
    return sorted(set(result), key=lambda path: path.relative_to(root).as_posix())


def check_bytes(relative: str, content: bytes) -> list[Finding]:
    if Path(relative).suffix.lower() in BINARY_SUFFIXES:
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(relative, "non-UTF-8 content outside approved media")]
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        if HOME_PATH.search(line):
            findings.append(Finding(relative, "machine-specific user-home path", number))
        if SECRET.search(line):
            findings.append(Finding(relative, "possible credential/private key; review without logging its value", number))
        for match in IPV4.finditer(line):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if any(address in network for network in PRIVATE_NETWORKS):
                findings.append(Finding(relative, "private device IP; use an explicit user-supplied address", number))
                break
    return findings


def _git_entries(root: Path) -> dict[str, str]:
    result = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "-z"],
                            capture_output=True, check=True)
    entries = {}
    for value in result.stdout.split(b"\0"):
        if value:
            metadata, name = value.split(b"\t", 1)
            entries[name.decode("utf-8")] = metadata.split()[0].decode("ascii")
    return entries


def check_repository(root: Path, mode: str = "tracked") -> tuple[list[Finding], list[Path]]:
    root = Path(root).resolve()
    indexed = {} if mode == "source" else _git_entries(root)
    paths = public_source_files(root) if mode == "source" else [root / name for name in indexed]
    findings = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if not is_public_path(relative):
            findings.append(Finding(relative, "not in the public source allowlist (generated, private, or unreviewed path)"))
            continue
        if indexed.get(relative, "100644") not in {"100644", "100755"}:
            findings.append(Finding(relative, "indexed source must be a regular file, not a symlink/submodule"))
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            findings.append(Finding(relative, "symlinks are not included in public source exports"))
            continue
        if mode == "staged":
            result = subprocess.run(["git", "-C", str(root), "show", ":" + relative],
                                    capture_output=True, check=True)
            content = result.stdout
        elif path.is_file():
            content = path.read_bytes()
        else:
            findings.append(Finding(relative, "indexed source file is missing from the working tree"))
            continue
        if len(content) > 2_000_000:
            findings.append(Finding(relative, "source file exceeds 2 MB; review artifact placement"))
        findings.extend(check_bytes(relative, content))
    return findings, paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("tracked", "staged", "source"), default="tracked",
                        help="tracked working tree, complete staged index, or explicit public source selection")
    args = parser.parse_args(argv)
    try:
        findings, paths = check_repository(args.root, args.mode)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error("Could not read source/index; use --mode source before initializing Git. " + type(error).__name__)
    for finding in findings:
        print(finding.display())
    print(f"Checked {len(paths)} {args.mode} source files; {len(findings)} finding(s). Values are not printed.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
