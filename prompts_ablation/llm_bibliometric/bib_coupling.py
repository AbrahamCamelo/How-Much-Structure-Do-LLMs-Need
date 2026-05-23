from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix


REFERENCE_SEPARATOR = ";"


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


def parse_reference_sets(scopus_df: pd.DataFrame) -> dict[int, set[str]]:
    reference_sets: dict[int, set[str]] = {}
    for _, row in scopus_df.iterrows():
        paper_id = int(row["paper_id"])
        references = {
            reference.strip()
            for reference in str(row.get("references", "")).split(REFERENCE_SEPARATOR)
            if reference.strip()
        }
        reference_sets[paper_id] = references
    return reference_sets


def build_reference_matrix(
    scopus_df: pd.DataFrame,
) -> tuple[csr_matrix, list[int], dict[int, set[str]]]:
    paper_ids = scopus_df["paper_id"].astype(int).tolist()
    reference_sets = parse_reference_sets(scopus_df)
    all_references = sorted(
        {reference for references in reference_sets.values() for reference in references}
    )

    if not all_references:
        empty_matrix = csr_matrix((len(paper_ids), 0), dtype=np.float32)
        return empty_matrix, paper_ids, reference_sets

    reference_index = {reference: idx for idx, reference in enumerate(all_references)}
    matrix = lil_matrix((len(paper_ids), len(all_references)), dtype=np.float32)

    for row_index, paper_id in enumerate(paper_ids):
        for reference in reference_sets[paper_id]:
            matrix[row_index, reference_index[reference]] = 1.0
    return matrix.tocsr(), paper_ids, reference_sets


def compute_association_strength(reference_matrix: csr_matrix) -> np.ndarray:
    if reference_matrix.shape[1] == 0:
        return np.zeros((reference_matrix.shape[0], reference_matrix.shape[0]), dtype=np.float32)

    coupling = (reference_matrix @ reference_matrix.T).toarray().astype(np.float32)
    np.fill_diagonal(coupling, 0.0)

    strengths = np.asarray(reference_matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    denominator = np.outer(strengths, strengths)
    with np.errstate(divide="ignore", invalid="ignore"):
        association_strength = np.where(denominator > 0, coupling / denominator, 0.0)
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


def detect_louvain_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
) -> tuple[list[set[int]], dict[int, int]]:
    if graph.number_of_nodes() == 0:
        return [], {}

    raw_communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=42,
    )
    communities = [
        {int(node) for node in community}
        for community in raw_communities
        if len(community) >= min_cluster_size
    ]
    communities.sort(key=len, reverse=True)

    cluster_map: dict[int, int] = {}
    for cluster_id, community in enumerate(communities, start=1):
        for paper_id in community:
            cluster_map[paper_id] = cluster_id
    return communities, cluster_map


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

        if _cluster_probe_score(next_probe, target_cluster_count, start_resolution) < _cluster_probe_score(
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

            if _cluster_probe_score(mid_probe, target_cluster_count, start_resolution) < _cluster_probe_score(
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
    labeled["degree"] = labeled["paper_id"].map(degree_map).fillna(0).astype(int)
    labeled["cluster_size"] = labeled["cluster"].map(cluster_sizes).astype("Int64")
    return labeled


def run_bibliographic_coupling(
    scopus_df: pd.DataFrame,
    min_weight: float = 0.001,
    top_n_edges: int | None = None,
    min_degree: int = 1,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    target_cluster_count: int | None = None,
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
    if target_cluster_count is None:
        communities, cluster_map = detect_louvain_communities(
            graph=graph,
            resolution=resolution,
            min_cluster_size=min_cluster_size,
        )
        matched_target_exactly = False
        used_resolution = resolution
    else:
        search_result = search_resolution_for_target_clusters(
            graph=graph,
            target_cluster_count=target_cluster_count,
            start_resolution=resolution,
            initial_step=resolution_step,
            min_cluster_size=min_cluster_size,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            max_iterations=resolution_max_iterations,
            min_interval=resolution_min_interval,
        )
        communities = search_result.communities
        cluster_map = search_result.cluster_map
        matched_target_exactly = search_result.matched_target_exactly
        used_resolution = search_result.resolution
    if communities and graph.number_of_edges() > 0:
        clustered_nodes = set().union(*communities)
        modularity_graph = graph.subgraph(clustered_nodes).copy()
        modularity = float(
            nx.community.modularity(modularity_graph, communities, weight="weight")
        )
    else:
        modularity = float("nan")
    labeled = label_scopus_with_clusters(scopus_df, graph, cluster_map, communities)
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
    )


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
