"""Safely delete NIfTI scans listed under ``excluded nifti`` in a JSON file."""

import argparse
import json
from pathlib import Path


def is_beneath(path, root):
    """Return True when ``path`` resolves to ``root`` or one of its children."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete files listed in the 'excluded nifti' field. The default "
            "is a dry run; pass --execute to perform deletion."
        )
    )
    parser.add_argument("json_path", type=Path, help="Path to the NIfTI JSON file")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for relative paths and the allowed deletion root. "
            "Defaults to the JSON file's parent directory."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, only report what would happen.",
    )
    args = parser.parse_args()

    json_path = args.json_path.expanduser().resolve()
    base_dir = (
        args.base_dir.expanduser().resolve()
        if args.base_dir is not None
        else json_path.parent
    )

    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    excluded = data.get("excluded nifti")
    if not isinstance(excluded, list):
        raise ValueError("JSON field 'excluded nifti' must be a list")

    candidates = []
    invalid_extension = []
    outside_base_dir = []
    seen = set()

    for entry in excluded:
        if not isinstance(entry, str) or not entry.endswith(".nii.gz"):
            invalid_extension.append(entry)
            continue

        listed_path = Path(entry).expanduser()
        resolved_path = (
            listed_path.resolve()
            if listed_path.is_absolute()
            else (base_dir / listed_path).resolve()
        )
        if not is_beneath(resolved_path, base_dir):
            outside_base_dir.append(entry)
            continue
        if resolved_path not in seen:
            seen.add(resolved_path)
            candidates.append(resolved_path)

    existing = [path for path in candidates if path.is_file()]
    missing = [path for path in candidates if not path.exists()]
    non_files = [path for path in candidates if path.exists() and not path.is_file()]

    mode = "DELETE" if args.execute else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"JSON: {json_path}")
    print(f"Allowed root: {base_dir}")
    print(f"Excluded entries: {len(excluded)}")
    print(f"Unique valid paths: {len(candidates)}")
    print(f"Existing files: {len(existing)}")
    print(f"Missing paths: {len(missing)}")
    print(f"Non-file paths: {len(non_files)}")
    print(f"Rejected non-NIfTI entries: {len(invalid_extension)}")
    print(f"Rejected paths outside allowed root: {len(outside_base_dir)}")

    if outside_base_dir:
        print("Refusing to continue because the JSON contains paths outside the allowed root.")
        return 2

    if not args.execute:
        print("No files deleted. Re-run with --execute after reviewing this summary.")
        return 0

    deleted = 0
    errors = []
    for path in existing:
        try:
            path.unlink()
            deleted += 1
        except OSError as error:
            errors.append((path, error))

    print(f"Deleted files: {deleted}")
    print(f"Deletion errors: {len(errors)}")
    for path, error in errors:
        print(f"  ERROR: {path}: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
