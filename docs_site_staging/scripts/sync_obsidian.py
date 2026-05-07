from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML. Run `pip install -r requirements.txt` first."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "publish.yml"
TEXT_SUFFIXES = {".md", ".markdown"}
ASSET_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".bmp",
    ".pdf",
}


@dataclass(frozen=True)
class Entry:
    source: Path
    dest: Path
    enabled: bool


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if "source_root" not in data or "target_root" not in data:
        raise SystemExit("publish.yml must define source_root and target_root.")

    return data


def parse_entries(config: dict[str, Any]) -> tuple[Path, Path, list[Entry]]:
    source_root = Path(config["source_root"]).expanduser().resolve()
    target_root = (ROOT / config["target_root"]).resolve()
    entries: list[Entry] = []

    for raw in config.get("entries", []):
        source = (source_root / raw["source"]).resolve()
        dest = (target_root / raw.get("dest", raw["source"])).resolve()
        enabled = bool(raw.get("enabled", True))
        entries.append(Entry(source=source, dest=dest, enabled=enabled))

    return source_root, target_root, entries


def ensure_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to write outside target root: {path}") from exc


def slugify_path_part(part: str) -> str:
    return part.replace("\\", "/")


def iter_entry_files(entry: Entry) -> list[Path]:
    if entry.source.is_file():
        return [entry.source]
    return [p for p in entry.source.rglob("*") if p.is_file()]


def entry_dest_for_file(entry: Entry, source: Path) -> Path:
    if entry.source.is_file():
        return entry.dest
    rel = source.relative_to(entry.source)
    return entry.dest / rel


def build_publish_index(entries: list[Entry]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for entry in entries:
        if not entry.enabled or not entry.source.exists():
            continue
        for path in iter_entry_files(entry):
            if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix.lower() not in ASSET_SUFFIXES:
                continue
            rel = path.relative_to(entry.source.parent if entry.source.is_file() else entry.source)
            dest = entry_dest_for_file(entry, path)
            stem = path.stem
            index.setdefault(stem, dest)
            index.setdefault(path.name, dest)
            index.setdefault(slugify_path_part(str(rel.with_suffix(""))), dest)
            index.setdefault(slugify_path_part(str(rel)), dest)
    return index


def relative_link(current_dest: Path, target_dest: Path) -> str:
    from os.path import relpath

    link = relpath(target_dest, start=current_dest.parent)
    return link.replace("\\", "/")


def convert_wikilinks(text: str, current_dest: Path, publish_index: dict[str, Path]) -> str:
    def replace_embed(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        name, _, alias = target.partition("|")
        published = publish_index.get(name) or publish_index.get(f"{name}.md")
        if published:
            link = relative_link(current_dest, published)
            label = alias or Path(name).name
            if published.suffix.lower() in ASSET_SUFFIXES:
                return f"![{label}]({link})"
            return f"[{label}]({link})"

        label = alias or Path(name).name
        return f"![{label}]({name})"

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        name, _, alias = target.partition("|")
        heading = ""
        if "#" in name:
            name, _, heading = name.partition("#")

        published = publish_index.get(name) or publish_index.get(f"{name}.md")
        label = alias or heading or Path(name).name
        if published:
            link = relative_link(current_dest, published)
            if heading:
                link = f"{link}#{heading}"
            return f"[{label}]({link})"

        return f"[{label}]({name})"

    text = re.sub(r"!\[\[([^\]]+)\]\]", replace_embed, text)
    text = re.sub(r"(?<!!)\[\[([^\]]+)\]\]", replace_link, text)
    return text


def copy_file(source: Path, dest: Path, publish_index: dict[str, Path]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8")
        text = convert_wikilinks(text, dest, publish_index)
        with dest.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    else:
        shutil.copy2(source, dest)


def copy_entry(
    entry: Entry,
    target_root: Path,
    publish_index: dict[str, Path],
    dry_run: bool,
) -> list[str]:
    if not entry.source.exists():
        raise SystemExit(f"Configured source does not exist: {entry.source}")

    actions: list[str] = []
    files = iter_entry_files(entry)

    for source in files:
        dest = entry_dest_for_file(entry, source)
        ensure_inside(dest, target_root)
        actions.append(f"copy {source} -> {dest}")
        if not dry_run:
            copy_file(source, dest, publish_index)

    return actions


def reset_target(target_root: Path, dry_run: bool) -> None:
    ensure_inside(target_root, ROOT / "docs")
    if target_root.exists():
        if dry_run:
            print(f"would remove {target_root}")
        else:
            shutil.rmtree(target_root)
    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync selected Obsidian notes into docs/.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files.")
    args = parser.parse_args()

    config = load_config()
    source_root, target_root, entries = parse_entries(config)

    if not source_root.exists():
        raise SystemExit(f"Obsidian source root does not exist: {source_root}")

    reset_target(target_root, args.dry_run)

    enabled_entries = [entry for entry in entries if entry.enabled]
    disabled_entries = [entry for entry in entries if not entry.enabled]
    publish_index = build_publish_index(enabled_entries)

    print(f"source: {source_root}")
    print(f"target: {target_root}")
    print(f"enabled entries: {len(enabled_entries)}")
    print(f"disabled entries: {len(disabled_entries)}")

    for entry in enabled_entries:
        for action in copy_entry(entry, target_root, publish_index, args.dry_run):
            print(action)

    if disabled_entries:
        print("disabled:")
        for entry in disabled_entries:
            print(f"skip {entry.source}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
