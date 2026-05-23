from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

import networkx as nx
import numpy as np
import pandas as pd

from .constants import DEFAULT_LOUVAIN_RESOLUTION


MIN_TITLE_MATCH_ALNUM_CHARS = 24
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class BiblioCouplingResult:
    graph: nx.Graph
    similarity_matrix: np.ndarray
    labeled_scopus: pd.DataFrame
    communities: list[set[int]]
    modularity: float
    resolution: float
    cluster_count: int
    target_cluster_count: int | None
    matched_target_cluster_count: bool
    max_cluster_count: int | None
    min_cluster_size: int
    effective_min_cluster_size: int
    min_cluster_proportion: float | None
    min_cluster_proportion_basis: str | None
    cluster_selection_strategy: str | None


@dataclass(frozen=True)
class ResolutionProbe:
    resolution: float
    cluster_count: int


@dataclass(frozen=True)
class ResolutionSearchResult:
    resolution: float
    cluster_count: int
    communities: list[set[int]]
    cluster_map: dict[int, int]
    matched_target_exactly: bool
    history: tuple[ResolutionProbe, ...]


@dataclass(frozen=True)
class MaxClusterSearchResult:
    resolution: float
    cluster_count: int
    communities: list[set[int]]
    cluster_map: dict[int, int]
    satisfied_max_cluster_count: bool
    history: tuple[ResolutionProbe, ...]


def _normalize_communities(
    communities: list[set[int]],
) -> tuple[list[set[int]], dict[int, int]]:
    normalized = [{int(node) for node in community} for community in communities if community]
    normalized.sort(key=lambda community: (-len(community), min(community)))

    cluster_map: dict[int, int] = {}
    for cluster_id, community in enumerate(normalized, start=1):
        for paper_id in community:
            cluster_map[paper_id] = cluster_id
    return normalized, cluster_map


def _ascii_fold(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", "" if text is None else str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_reference_blob(text: object) -> str:
    lowered = _ascii_fold(text).lower()
    lowered = NON_ALNUM_PATTERN.sub(" ", lowered)
    return WHITESPACE_PATTERN.sub(" ", lowered).strip()


def _normalize_identifier(text: object) -> str:
    return WHITESPACE_PATTERN.sub(" ", _ascii_fold(text).lower()).strip()


def _build_direct_citation_candidates(scopus_df: pd.DataFrame) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for _, row in scopus_df.iterrows():
        paper_id = int(row["paper_id"])
        title_key = _normalize_reference_blob(row.get("paper_title", ""))
        title_alnum_length = len(title_key.replace(" ", ""))
        year_value = pd.to_numeric(pd.Series([row.get("year")]), errors="coerce").iloc[0]
        year_text = str(int(year_value)) if pd.notna(year_value) else ""
        candidates.append(
            {
                "paper_id": paper_id,
                "title_key": title_key,
                "title_alnum_length": title_alnum_length,
                "year_text": year_text,
                "doi_key": _normalize_identifier(row.get("doi", "")),
                "eid_key": _normalize_identifier(row.get("eid", "")),
            }
        )
    return candidates


def resolve_direct_citation_links(scopus_df: pd.DataFrame) -> dict[int, set[int]]:
    candidates = _build_direct_citation_candidates(scopus_df)
    citation_links: dict[int, set[int]] = {}

    for _, row in scopus_df.iterrows():
        citing_paper_id = int(row["paper_id"])
        references_raw = "" if pd.isna(row.get("references")) else str(row.get("references", ""))
        references_blob = _normalize_reference_blob(references_raw)
        references_raw_key = _normalize_identifier(references_raw)
        cited_paper_ids: set[int] = set()

        for candidate in candidates:
            cited_paper_id = int(candidate["paper_id"])
            if cited_paper_id == citing_paper_id:
                continue

            doi_key = str(candidate["doi_key"])
            if doi_key and doi_key != "nan" and doi_key in references_raw_key:
                cited_paper_ids.add(cited_paper_id)
                continue

            eid_key = str(candidate["eid_key"])
            if eid_key and eid_key != "nan" and eid_key in references_raw_key:
                cited_paper_ids.add(cited_paper_id)
                continue

            title_key = str(candidate["title_key"])
            if not title_key:
                continue
            if int(candidate["title_alnum_length"]) < MIN_TITLE_MATCH_ALNUM_CHARS:
                continue

            year_text = str(candidate["year_text"])
            if year_text and year_text not in references_raw_key:
                continue

            if title_key in references_blob:
                cited_paper_ids.add(cited_paper_id)

        citation_links[citing_paper_id] = cited_paper_ids

    return citation_links


def build_reference_matrix(
    scopus_df: pd.DataFrame,
) -> tuple[np.ndarray, list[int], dict[int, set[int]]]:
    paper_ids = scopus_df["paper_id"].astype(int).tolist()
    paper_index = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    citation_links = resolve_direct_citation_links(scopus_df)
    direct_citation_matrix = np.zeros((len(paper_ids), len(paper_ids)), dtype=np.float32)

    for citing_paper_id, cited_paper_ids in citation_links.items():
        source_index = paper_index[citing_paper_id]
        for cited_paper_id in cited_paper_ids:
            target_index = paper_index.get(cited_paper_id)
            if target_index is None or target_index == source_index:
                continue
            direct_citation_matrix[source_index, target_index] += 1.0

    return direct_citation_matrix, paper_ids, citation_links


def compute_association_strength(reference_matrix: np.ndarray) -> np.ndarray:
    if reference_matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    association_strength = (reference_matrix + reference_matrix.T).astype(np.float32)
    np.fill_diagonal(association_strength, 0.0)
    return association_strength


def build_similarity_graph(
    paper_ids: list[int],
    association_strength: np.ndarray,
    min_weight: float = 0.001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
) -> nx.Graph:
    edges: list[tuple[int, int, float]] = []
    for i, left_paper_id in enumerate(paper_ids):
        for j in range(i + 1, len(paper_ids)):
            weight = float(association_strength[i, j])
            if weight >= min_weight:
                edges.append((left_paper_id, paper_ids[j], weight))

    edges.sort(key=lambda item: item[2], reverse=True)
    if top_n_edges is not None:
        edges = edges[:top_n_edges]

    graph = nx.Graph()
    graph.add_nodes_from(paper_ids)
    for left_paper_id, right_paper_id, weight in edges:
        graph.add_edge(left_paper_id, right_paper_id, weight=weight)

    weak_nodes = [node for node, degree in graph.degree() if degree < min_degree]
    if weak_nodes:
        graph.remove_nodes_from(weak_nodes)

    return graph


def _run_louvain_raw_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
) -> list[set[int]]:
    if graph.number_of_nodes() == 0:
        return []

    raw_communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=42,
    )
    return [
        {int(node) for node in community}
        for community in raw_communities
        if community
    ]


def resolve_effective_min_cluster_size(
    raw_communities: list[set[int]],
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
) -> tuple[int, str | None]:
    effective_min_cluster_size = max(int(min_cluster_size), 1)
    if min_cluster_proportion is None:
        return effective_min_cluster_size, None

    proportion = float(min_cluster_proportion)
    if proportion < 0.0 or proportion > 1.0:
        raise ValueError("min_cluster_proportion must be between 0.0 and 1.0.")

    largest_cluster_size = max((len(community) for community in raw_communities), default=0)
    proportional_floor = int(math.ceil(float(largest_cluster_size) * proportion))
    return max(effective_min_cluster_size, proportional_floor), "largest_cluster"


def detect_louvain_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
) -> tuple[list[set[int]], dict[int, int]]:
    communities, cluster_map, _, _ = detect_louvain_communities_detailed(
        graph=graph,
        resolution=resolution,
        min_cluster_size=min_cluster_size,
        min_cluster_proportion=min_cluster_proportion,
    )
    return communities, cluster_map


def detect_louvain_communities_detailed(
    graph: nx.Graph,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
) -> tuple[list[set[int]], dict[int, int], int, str | None]:
    raw_communities = _run_louvain_raw_communities(
        graph=graph,
        resolution=resolution,
    )
    effective_min_cluster_size, min_cluster_proportion_basis = (
        resolve_effective_min_cluster_size(
            raw_communities=raw_communities,
            min_cluster_size=min_cluster_size,
            min_cluster_proportion=min_cluster_proportion,
        )
    )
    communities = [
        community
        for community in raw_communities
        if len(community) >= effective_min_cluster_size
    ]
    normalized_communities, cluster_map = _normalize_communities(communities)
    return (
        normalized_communities,
        cluster_map,
        effective_min_cluster_size,
        min_cluster_proportion_basis,
    )


def select_top_communities(
    communities: list[set[int]],
    max_cluster_count: int | None,
) -> tuple[list[set[int]], dict[int, int], str | None]:
    normalized_communities, _ = _normalize_communities(communities)
    if max_cluster_count is None or len(normalized_communities) <= max_cluster_count:
        cluster_map = {}
        for cluster_id, community in enumerate(normalized_communities, start=1):
            for paper_id in community:
                cluster_map[paper_id] = cluster_id
        return normalized_communities, cluster_map, None

    selected_communities = normalized_communities[: int(max_cluster_count)]
    selected_communities, cluster_map = _normalize_communities(selected_communities)
    return selected_communities, cluster_map, "largest_clusters"


def _cluster_probe_score(
    probe: ResolutionProbe,
    target_cluster_count: int,
    start_resolution: float,
) -> tuple[float, float]:
    return (
        abs(probe.cluster_count - target_cluster_count),
        abs(probe.resolution - start_resolution),
    )


def search_resolution_for_target_clusters(
    graph: nx.Graph,
    target_cluster_count: int,
    start_resolution: float = 1.0,
    initial_step: float = 0.1,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    max_iterations: int = 50,
    min_interval: float = 0.001,
) -> ResolutionSearchResult:
    if target_cluster_count < 1:
        raise ValueError("target_cluster_count must be at least 1.")

    if graph.number_of_nodes() == 0:
        return ResolutionSearchResult(
            resolution=start_resolution,
            cluster_count=0,
            communities=[],
            cluster_map={},
            matched_target_exactly=False,
            history=tuple(),
        )

    probe_cache: dict[float, tuple[ResolutionProbe, list[set[int]], dict[int, int]]] = {}

    def evaluate(resolution: float) -> tuple[ResolutionProbe, list[set[int]], dict[int, int]]:
        normalized_resolution = round(
            min(max(float(resolution), min_resolution), max_resolution),
            10,
        )
        cached = probe_cache.get(normalized_resolution)
        if cached is not None:
            return cached
        communities, cluster_map = detect_louvain_communities(
            graph=graph,
            resolution=normalized_resolution,
            min_cluster_size=min_cluster_size,
            min_cluster_proportion=min_cluster_proportion,
        )
        probe = ResolutionProbe(
            resolution=normalized_resolution,
            cluster_count=len(communities),
        )
        record = (probe, communities, cluster_map)
        probe_cache[normalized_resolution] = record
        return record

    best_probe, best_communities, best_cluster_map = evaluate(start_resolution)
    history: list[ResolutionProbe] = [best_probe]
    if best_probe.cluster_count == target_cluster_count:
        return ResolutionSearchResult(
            resolution=best_probe.resolution,
            cluster_count=best_probe.cluster_count,
            communities=best_communities,
            cluster_map=best_cluster_map,
            matched_target_exactly=True,
            history=tuple(history),
        )

    direction = 1.0 if best_probe.cluster_count < target_cluster_count else -1.0
    current_probe, current_communities, current_cluster_map = (
        best_probe,
        best_communities,
        best_cluster_map,
    )
    lower_probe: ResolutionProbe | None = None
    upper_probe: ResolutionProbe | None = None

    for _ in range(max_iterations):
        next_resolution = current_probe.resolution + (direction * initial_step)
        next_resolution = min(max(next_resolution, min_resolution), max_resolution)
        if next_resolution == current_probe.resolution:
            break

        next_probe, next_communities, next_cluster_map = evaluate(next_resolution)
        if next_probe.resolution not in {probe.resolution for probe in history}:
            history.append(next_probe)

        if _cluster_probe_score(
            next_probe,
            target_cluster_count,
            start_resolution,
        ) < _cluster_probe_score(
            best_probe,
            target_cluster_count,
            start_resolution,
        ):
            best_probe, best_communities, best_cluster_map = (
                next_probe,
                next_communities,
                next_cluster_map,
            )

        if next_probe.cluster_count == target_cluster_count:
            return ResolutionSearchResult(
                resolution=next_probe.resolution,
                cluster_count=next_probe.cluster_count,
                communities=next_communities,
                cluster_map=next_cluster_map,
                matched_target_exactly=True,
                history=tuple(history),
            )

        crossed_target = (
            current_probe.cluster_count < target_cluster_count < next_probe.cluster_count
        ) or (
            current_probe.cluster_count > target_cluster_count > next_probe.cluster_count
        )
        if crossed_target:
            lower_probe = (
                current_probe if current_probe.cluster_count < target_cluster_count else next_probe
            )
            upper_probe = (
                current_probe if current_probe.cluster_count > target_cluster_count else next_probe
            )
            break

        current_probe, current_communities, current_cluster_map = (
            next_probe,
            next_communities,
            next_cluster_map,
        )

    if lower_probe is not None and upper_probe is not None:
        for _ in range(max_iterations):
            if (upper_probe.resolution - lower_probe.resolution) <= min_interval:
                break
            midpoint = round((lower_probe.resolution + upper_probe.resolution) / 2.0, 10)
            mid_probe, mid_communities, mid_cluster_map = evaluate(midpoint)
            if mid_probe.resolution not in {probe.resolution for probe in history}:
                history.append(mid_probe)

            if _cluster_probe_score(
                mid_probe,
                target_cluster_count,
                start_resolution,
            ) < _cluster_probe_score(
                best_probe,
                target_cluster_count,
                start_resolution,
            ):
                best_probe, best_communities, best_cluster_map = (
                    mid_probe,
                    mid_communities,
                    mid_cluster_map,
                )

            if mid_probe.cluster_count == target_cluster_count:
                return ResolutionSearchResult(
                    resolution=mid_probe.resolution,
                    cluster_count=mid_probe.cluster_count,
                    communities=mid_communities,
                    cluster_map=mid_cluster_map,
                    matched_target_exactly=True,
                    history=tuple(history),
                )

            if mid_probe.cluster_count < target_cluster_count:
                lower_probe = mid_probe
            elif mid_probe.cluster_count > target_cluster_count:
                upper_probe = mid_probe
            else:
                return ResolutionSearchResult(
                    resolution=mid_probe.resolution,
                    cluster_count=mid_probe.cluster_count,
                    communities=mid_communities,
                    cluster_map=mid_cluster_map,
                    matched_target_exactly=True,
                    history=tuple(history),
                )

    return ResolutionSearchResult(
        resolution=best_probe.resolution,
        cluster_count=best_probe.cluster_count,
        communities=best_communities,
        cluster_map=best_cluster_map,
        matched_target_exactly=(best_probe.cluster_count == target_cluster_count),
        history=tuple(history),
    )


def search_resolution_for_max_clusters(
    graph: nx.Graph,
    max_cluster_count: int,
    start_resolution: float = 1.0,
    initial_step: float = 0.1,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
    min_resolution: float = 0.05,
    max_iterations: int = 50,
) -> MaxClusterSearchResult:
    if max_cluster_count < 1:
        raise ValueError("max_cluster_count must be at least 1.")
    if initial_step <= 0:
        raise ValueError("initial_step must be greater than 0.")

    if graph.number_of_nodes() == 0:
        return MaxClusterSearchResult(
            resolution=start_resolution,
            cluster_count=0,
            communities=[],
            cluster_map={},
            satisfied_max_cluster_count=True,
            history=tuple(),
        )

    probe_cache: dict[float, tuple[ResolutionProbe, list[set[int]], dict[int, int]]] = {}

    def evaluate(resolution: float) -> tuple[ResolutionProbe, list[set[int]], dict[int, int]]:
        normalized_resolution = round(max(float(resolution), min_resolution), 10)
        cached = probe_cache.get(normalized_resolution)
        if cached is not None:
            return cached
        communities, cluster_map = detect_louvain_communities(
            graph=graph,
            resolution=normalized_resolution,
            min_cluster_size=min_cluster_size,
            min_cluster_proportion=min_cluster_proportion,
        )
        probe = ResolutionProbe(
            resolution=normalized_resolution,
            cluster_count=len(communities),
        )
        record = (probe, communities, cluster_map)
        probe_cache[normalized_resolution] = record
        return record

    current_probe, current_communities, current_cluster_map = evaluate(start_resolution)
    history: list[ResolutionProbe] = [current_probe]
    best_probe, best_communities, best_cluster_map = (
        current_probe,
        current_communities,
        current_cluster_map,
    )
    if current_probe.cluster_count <= max_cluster_count:
        return MaxClusterSearchResult(
            resolution=current_probe.resolution,
            cluster_count=current_probe.cluster_count,
            communities=current_communities,
            cluster_map=current_cluster_map,
            satisfied_max_cluster_count=True,
            history=tuple(history),
        )

    for _ in range(max_iterations):
        next_resolution = round(
            max(current_probe.resolution - initial_step, min_resolution),
            10,
        )
        if next_resolution == current_probe.resolution:
            break

        next_probe, next_communities, next_cluster_map = evaluate(next_resolution)
        if next_probe.resolution not in {probe.resolution for probe in history}:
            history.append(next_probe)

        if (
            next_probe.cluster_count < best_probe.cluster_count
            or (
                next_probe.cluster_count == best_probe.cluster_count
                and abs(next_probe.resolution - start_resolution)
                < abs(best_probe.resolution - start_resolution)
            )
        ):
            best_probe, best_communities, best_cluster_map = (
                next_probe,
                next_communities,
                next_cluster_map,
            )

        if next_probe.cluster_count <= max_cluster_count:
            return MaxClusterSearchResult(
                resolution=next_probe.resolution,
                cluster_count=next_probe.cluster_count,
                communities=next_communities,
                cluster_map=next_cluster_map,
                satisfied_max_cluster_count=True,
                history=tuple(history),
            )

        current_probe, current_communities, current_cluster_map = (
            next_probe,
            next_communities,
            next_cluster_map,
        )

    return MaxClusterSearchResult(
        resolution=best_probe.resolution,
        cluster_count=best_probe.cluster_count,
        communities=best_communities,
        cluster_map=best_cluster_map,
        satisfied_max_cluster_count=(best_probe.cluster_count <= max_cluster_count),
        history=tuple(history),
    )


def _inter_community_weight(
    graph: nx.Graph,
    left: set[int],
    right: set[int],
) -> float:
    right_nodes = set(right)
    total_weight = 0.0
    for node in left:
        for neighbor, edge_data in graph[node].items():
            if neighbor in right_nodes:
                total_weight += float(edge_data.get("weight", 1.0))
    return total_weight


def _merge_communities_to_target(
    graph: nx.Graph,
    communities: list[set[int]],
    target_cluster_count: int,
) -> tuple[list[set[int]], dict[int, int]]:
    working = [set(community) for community in communities if community]
    while len(working) > target_cluster_count:
        best_pair: tuple[int, int] | None = None
        best_weight = float("-inf")
        best_combined_size: int | None = None
        best_size_gap: int | None = None

        for left_index in range(len(working)):
            for right_index in range(left_index + 1, len(working)):
                left = working[left_index]
                right = working[right_index]
                weight = _inter_community_weight(graph, left, right)
                combined_size = len(left) + len(right)
                size_gap = abs(len(left) - len(right))

                if (
                    best_pair is None
                    or weight > best_weight
                    or (
                        weight == best_weight
                        and combined_size < (best_combined_size if best_combined_size is not None else combined_size + 1)
                    )
                    or (
                        weight == best_weight
                        and combined_size == best_combined_size
                        and size_gap < (best_size_gap if best_size_gap is not None else size_gap + 1)
                    )
                ):
                    best_pair = (left_index, right_index)
                    best_weight = weight
                    best_combined_size = combined_size
                    best_size_gap = size_gap

        if best_pair is None:
            break

        left_index, right_index = best_pair
        merged = working[left_index] | working[right_index]
        remaining = [
            community
            for index, community in enumerate(working)
            if index not in {left_index, right_index}
        ]
        remaining.append(merged)
        working = remaining

    return _normalize_communities(working)


def _split_community_into_two(
    graph: nx.Graph,
    community: set[int],
    min_cluster_size: int,
    resolution: float,
    max_resolution: float = 5.0,
    resolution_step: float = 0.25,
) -> list[set[int]] | None:
    subgraph = graph.subgraph(community).copy()
    if subgraph.number_of_nodes() < max(2, min_cluster_size * 2):
        return None

    split_resolution = max(float(resolution), 1.0)
    while split_resolution <= max_resolution:
        raw_communities = nx.community.louvain_communities(
            subgraph,
            weight="weight",
            resolution=split_resolution,
            seed=42,
        )
        valid = [
            {int(node) for node in subcommunity}
            for subcommunity in raw_communities
            if len(subcommunity) >= min_cluster_size
        ]
        if len(valid) >= 2:
            valid, _ = _normalize_communities(valid)
            first = valid[0]
            second = set().union(*valid[1:])
            if len(first) >= min_cluster_size and len(second) >= min_cluster_size:
                return [first, second]
        split_resolution = round(split_resolution + resolution_step, 10)

    connected_parts = [
        {int(node) for node in component}
        for component in nx.connected_components(subgraph)
        if len(component) >= min_cluster_size
    ]
    if len(connected_parts) >= 2:
        connected_parts, _ = _normalize_communities(connected_parts)
        first = connected_parts[0]
        second = set().union(*connected_parts[1:])
        if len(first) >= min_cluster_size and len(second) >= min_cluster_size:
            return [first, second]

    weighted_degree = dict(subgraph.degree(weight="weight"))
    ordered_nodes = sorted(
        community,
        key=lambda node: (float(weighted_degree.get(node, 0.0)), int(node)),
        reverse=True,
    )
    midpoint = len(ordered_nodes) // 2
    left = set(ordered_nodes[:midpoint])
    right = set(ordered_nodes[midpoint:])
    if len(left) >= min_cluster_size and len(right) >= min_cluster_size:
        return [left, right]
    return None


def adjust_communities_to_target_count(
    graph: nx.Graph,
    communities: list[set[int]],
    target_cluster_count: int,
    min_cluster_size: int,
    resolution: float,
) -> tuple[list[set[int]], dict[int, int]]:
    adjusted, _ = _normalize_communities(communities)
    if len(adjusted) == target_cluster_count:
        return _normalize_communities(adjusted)

    if len(adjusted) > target_cluster_count:
        return _merge_communities_to_target(graph, adjusted, target_cluster_count)

    while len(adjusted) < target_cluster_count:
        split_index: int | None = None
        split_parts: list[set[int]] | None = None
        for index, community in enumerate(adjusted):
            candidate_parts = _split_community_into_two(
                graph=graph,
                community=community,
                min_cluster_size=min_cluster_size,
                resolution=resolution,
            )
            if candidate_parts is not None:
                split_index = index
                split_parts = candidate_parts
                break

        if split_index is None or split_parts is None:
            break

        adjusted = (
            adjusted[:split_index]
            + split_parts
            + adjusted[split_index + 1 :]
        )
        adjusted, _ = _normalize_communities(adjusted)

    if len(adjusted) > target_cluster_count:
        return _merge_communities_to_target(graph, adjusted, target_cluster_count)
    return _normalize_communities(adjusted)


def compute_link_strength(graph: nx.Graph) -> dict[int, float]:
    return {
        int(node): float(sum(data["weight"] for _, _, data in graph.edges(node, data=True)))
        for node in graph.nodes()
    }


def label_scopus_with_clusters(
    scopus_df: pd.DataFrame,
    graph: nx.Graph,
    cluster_map: dict[int, int],
    communities: list[set[int]],
) -> pd.DataFrame:
    labeled = scopus_df.copy()
    link_strength = compute_link_strength(graph)
    degree_map = dict(graph.degree())
    cluster_sizes = {index + 1: len(community) for index, community in enumerate(communities)}

    labeled["cluster"] = labeled["paper_id"].map(cluster_map).astype("Int64")
    labeled["link_strength"] = labeled["paper_id"].map(link_strength).fillna(0.0)
    labeled["clustering_method"] = "direct_citation"
    labeled["degree"] = labeled["paper_id"].map(degree_map).fillna(0).astype(int)
    labeled["cluster_size"] = labeled["cluster"].map(cluster_sizes).astype("Int64")
    return labeled


def run_bibliographic_coupling(
    scopus_df: pd.DataFrame,
    min_weight: float = 0.001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = DEFAULT_LOUVAIN_RESOLUTION,
    min_cluster_size: int = 3,
    min_cluster_proportion: float | None = None,
    target_cluster_count: int | None = None,
    max_cluster_count: int | None = None,
    resolution_step: float = 0.1,
    min_resolution: float = 0.05,
    max_resolution: float = 5.0,
    resolution_max_iterations: int = 50,
    resolution_min_interval: float = 0.001,
) -> BiblioCouplingResult:
    reference_matrix, paper_ids, _ = build_reference_matrix(scopus_df)
    association_strength = compute_association_strength(reference_matrix)
    graph = build_similarity_graph(
        paper_ids=paper_ids,
        association_strength=association_strength,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
    )
    requested_resolution = float(resolution)
    if target_cluster_count is None:
        communities, cluster_map, effective_min_cluster_size, min_cluster_proportion_basis = (
            detect_louvain_communities_detailed(
                graph=graph,
                resolution=requested_resolution,
                min_cluster_size=min_cluster_size,
                min_cluster_proportion=min_cluster_proportion,
            )
        )
        communities, cluster_map, cluster_selection_strategy = select_top_communities(
            communities=communities,
            max_cluster_count=max_cluster_count,
        )
        used_resolution = requested_resolution
        matched_target_exactly = False
    else:
        search_result = search_resolution_for_target_clusters(
            graph=graph,
            target_cluster_count=target_cluster_count,
            start_resolution=requested_resolution,
            initial_step=resolution_step,
            min_cluster_size=min_cluster_size,
            min_cluster_proportion=min_cluster_proportion,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            max_iterations=resolution_max_iterations,
            min_interval=resolution_min_interval,
        )
        communities = search_result.communities
        cluster_map = search_result.cluster_map
        matched_target_exactly = search_result.matched_target_exactly
        used_resolution = search_result.resolution
        (
            _,
            _,
            effective_min_cluster_size,
            min_cluster_proportion_basis,
        ) = detect_louvain_communities_detailed(
            graph=graph,
            resolution=used_resolution,
            min_cluster_size=min_cluster_size,
            min_cluster_proportion=min_cluster_proportion,
        )
        cluster_selection_strategy = None
        if not matched_target_exactly:
            communities, cluster_map = adjust_communities_to_target_count(
                graph=graph,
                communities=communities,
                target_cluster_count=target_cluster_count,
                min_cluster_size=effective_min_cluster_size,
                resolution=used_resolution,
            )
    if target_cluster_count is None:
        pass
    else:
        cluster_selection_strategy = None
    if communities and graph.number_of_edges() > 0:
        clustered_nodes = set().union(*communities)
        modularity_graph = graph.subgraph(clustered_nodes).copy()
        modularity = float(
            nx.community.modularity(modularity_graph, communities, weight="weight")
        )
    else:
        modularity = float("nan")
    labeled = label_scopus_with_clusters(scopus_df, graph, cluster_map, communities)
    labeled["clustering_min_cluster_size"] = int(min_cluster_size)
    labeled["clustering_effective_min_cluster_size"] = int(effective_min_cluster_size)
    labeled["clustering_min_cluster_proportion"] = (
        pd.Series([float(min_cluster_proportion)] * len(labeled), dtype="Float64")
        if min_cluster_proportion is not None
        else pd.Series([pd.NA] * len(labeled), dtype="Float64")
    )
    labeled["clustering_min_cluster_proportion_basis"] = (
        str(min_cluster_proportion_basis)
        if min_cluster_proportion_basis is not None
        else pd.NA
    )
    labeled["clustering_cluster_selection_strategy"] = (
        str(cluster_selection_strategy)
        if cluster_selection_strategy is not None
        else pd.NA
    )
    labeled["clustering_requested_resolution"] = requested_resolution
    labeled["clustering_resolution"] = float(used_resolution)
    labeled["clustering_max_clusters"] = (
        pd.Series([int(max_cluster_count)] * len(labeled), dtype="Int64")
        if max_cluster_count is not None
        else pd.Series([pd.NA] * len(labeled), dtype="Int64")
    )
    return BiblioCouplingResult(
        graph=graph,
        similarity_matrix=association_strength,
        labeled_scopus=labeled,
        communities=communities,
        modularity=modularity,
        resolution=float(used_resolution),
        cluster_count=len(communities),
        target_cluster_count=target_cluster_count,
        matched_target_cluster_count=matched_target_exactly,
        max_cluster_count=max_cluster_count,
        min_cluster_size=int(min_cluster_size),
        effective_min_cluster_size=effective_min_cluster_size,
        min_cluster_proportion=min_cluster_proportion,
        min_cluster_proportion_basis=min_cluster_proportion_basis,
        cluster_selection_strategy=cluster_selection_strategy,
    )


def run_direct_citation_clustering(*args, **kwargs) -> BiblioCouplingResult:
    return run_bibliographic_coupling(*args, **kwargs)


def select_top_clusters_by_link_strength(
    labeled_scopus_df: pd.DataFrame,
    top_k_clusters: int = 20,
) -> list[int]:
    clustered = labeled_scopus_df.dropna(subset=["cluster"]).copy()
    if clustered.empty:
        return []

    ranked = (
        clustered.groupby("cluster", dropna=True)
        .agg(
            total_link_strength=("link_strength", "sum"),
            cluster_size=("paper_id", "count"),
        )
        .sort_values(
            ["total_link_strength", "cluster_size"],
            ascending=[False, False],
        )
        .head(top_k_clusters)
    )
    return [int(cluster_id) for cluster_id in ranked.index.tolist()]


def compute_modularity_for_clusters(
    scopus_df: pd.DataFrame,
    cluster_column: str = "cluster",
    min_weight: float = 0.001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
) -> float:
    result = run_bibliographic_coupling(
        scopus_df=scopus_df,
        min_weight=min_weight,
        top_n_edges=top_n_edges,
        min_degree=min_degree,
        resolution=1.0,
        min_cluster_size=1,
    )
    graph = result.graph
    if graph.number_of_edges() == 0:
        return float("nan")

    clustered = scopus_df.dropna(subset=[cluster_column]).copy()
    communities: list[set[int]] = []
    for _, group in clustered.groupby(cluster_column):
        nodes = {
            int(paper_id)
            for paper_id in group["paper_id"].tolist()
            if graph.has_node(int(paper_id))
        }
        if nodes:
            communities.append(nodes)

    if not communities:
        return float("nan")
    clustered_nodes = set().union(*communities)
    modularity_graph = graph.subgraph(clustered_nodes).copy()
    return float(nx.community.modularity(modularity_graph, communities, weight="weight"))
