from __future__ import annotations

from pathlib import Path

from .constants import CLUSTERS_DIR, DESCRIPTION_DIR, EVALUATIONS_DIR, PROJECT_ROOT, QUALITY_DIR, SCOPUS_DIR


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def _resolve_exact_path(path_reference: str | Path) -> Path | None:
    path = Path(path_reference)
    if path.exists():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    return None


def _search_in_directories(file_name: str, directories: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for directory in directories:
        candidate = directory / file_name
        if candidate.exists():
            matches.append(candidate)
    return _deduplicate_paths(matches)


def _search_recursively(file_name: str, directories: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for directory in directories:
        if directory.exists():
            matches.extend(path for path in directory.rglob(file_name) if path.is_file())
    return _deduplicate_paths(matches)


def _resolve_unique_match(file_reference: str | Path, matches: list[Path], label: str) -> Path:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not resolve {label}: {file_reference}")
    options = ", ".join(str(match) for match in matches)
    raise ValueError(f"Ambiguous {label} '{file_reference}'. Matches: {options}")


def resolve_raw_scopus_file(file_reference: str | Path) -> Path:
    exact = _resolve_exact_path(file_reference)
    if exact is not None:
        return exact

    path = Path(file_reference)
    names = [path.name]
    if path.suffix == "":
        names.append(f"{path.name}.csv")

    matches: list[Path] = []
    for name in names:
        matches.extend(_search_in_directories(name, [SCOPUS_DIR]))
    return _resolve_unique_match(file_reference, _deduplicate_paths(matches), "Scopus file")


def resolve_labeled_scopus_file(file_reference: str | Path) -> Path:
    exact = _resolve_exact_path(file_reference)
    if exact is not None:
        return exact

    path = Path(file_reference)
    suffix = path.suffix or ".csv"
    names = [path.name if path.suffix else f"{path.name}{suffix}"]
    if not path.stem.endswith("_louvain"):
        names.insert(0, f"{path.stem}_louvain{suffix}")

    matches: list[Path] = []
    for name in names:
        matches.extend(_search_in_directories(name, [CLUSTERS_DIR]))
    return _resolve_unique_match(file_reference, _deduplicate_paths(matches), "clustered Scopus file")


def resolve_description_file(file_reference: str | Path) -> Path:
    exact = _resolve_exact_path(file_reference)
    if exact is not None:
        return exact

    path = Path(file_reference)
    names = [path.name]
    if path.suffix == "":
        names.append(f"{path.name}.csv")

    matches: list[Path] = []
    for name in names:
        matches.extend(_search_recursively(name, [DESCRIPTION_DIR]))
    return _resolve_unique_match(file_reference, _deduplicate_paths(matches), "description file")


def resolve_clustering_file(file_reference: str | Path) -> Path:
    exact = _resolve_exact_path(file_reference)
    if exact is not None:
        return exact

    path = Path(file_reference)
    suffix = path.suffix or ".csv"
    names = [path.name if path.suffix else f"{path.name}{suffix}"]
    if not path.stem.endswith("_louvain"):
        names.append(f"{path.stem}_louvain{suffix}")

    matches: list[Path] = []
    for name in names:
        matches.extend(_search_in_directories(name, [CLUSTERS_DIR]))
        matches.extend(_search_recursively(name, [QUALITY_DIR / "results", EVALUATIONS_DIR]))
    return _resolve_unique_match(file_reference, _deduplicate_paths(matches), "clustering file")


def resolve_output_file(file_reference: str | Path, default_directory: str | Path) -> Path:
    path = Path(file_reference)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return Path(default_directory) / path.name
