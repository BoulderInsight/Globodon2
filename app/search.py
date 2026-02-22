"""Core RAG search — embed query, cosine search, retrieve context."""

from collections import Counter

import numpy as np

from app.indexer import index
from app.llm import embed_text
from app.models import (
    MAFAction, MatchedCluster, RelatedPart, SearchResponse, SimilarTAR,
)


def search_tar(text: str, top_k: int = 10) -> SearchResponse:
    """Full RAG search pipeline: embed → cosine → cluster → MAF → parts."""

    # 1. Embed the query
    query_vec = embed_text(text)

    # 2. Cosine similarity (embeddings already normalized)
    sims = index.embeddings @ query_vec  # (n,)
    top_indices = np.argsort(sims)[::-1][:top_k]

    # 3. Build similar TARs list and collect cluster votes
    similar_tars: list[SimilarTAR] = []
    cluster_votes: list[int] = []
    all_part_numbers: set[str] = set()

    for idx in top_indices:
        row = index.tar_df.iloc[idx]
        sim_score = float(sims[idx])
        jcn = str(row.get("jcn", "")).strip()
        cluster_idx = int(index.tar_cluster_ids[idx])

        # Cluster label
        cluster_label = ""
        if 0 <= cluster_idx < len(index.cluster_profiles):
            cluster_label = index.cluster_profiles[cluster_idx]["problem"]
            cluster_votes.append(cluster_idx)

        # MAF actions for this TAR
        maf_records = index.maf_index.get(jcn, [])
        maf_actions = [
            MAFAction(
                corr_act=m["corr_act"],
                action_taken=m["action_taken"],
                manhours=m["manhours"],
                inst_partno=m["inst_partno"],
                rmvd_partno=m["rmvd_partno"],
            )
            for m in maf_records
        ]

        # Collect part numbers
        part_no = str(row.get("part_number", "")).strip()
        if part_no and part_no.lower() not in ("", "nan", "n/a", "none"):
            all_part_numbers.add(part_no)
        for m in maf_records:
            for key in ("inst_partno", "rmvd_partno"):
                pn = m[key]
                if pn and pn.lower() not in ("", "nan", "n/a", "none"):
                    all_part_numbers.add(pn)

        similar_tars.append(SimilarTAR(
            jcn=jcn,
            subject=str(row.get("subject", "")).strip(),
            issue=str(row.get("issue", "")).strip(),
            similarity=round(sim_score, 4),
            cluster_label=cluster_label,
            maf_actions=maf_actions,
            buno=str(row.get("buno", "")).strip(),
            activity=str(row.get("activity", "")).strip(),
            submit_date=str(row.get("submit_date", "")).strip(),
            status=str(row.get("status", "")).strip(),
            priority=str(row.get("priority", "")).strip(),
        ))

    # 4. Majority vote for best-match cluster
    matched_cluster = None
    if cluster_votes:
        vote_counts = Counter(cluster_votes)
        best_cluster_idx, best_count = vote_counts.most_common(1)[0]
        confidence = best_count / len(cluster_votes)
        profile = index.cluster_profiles[best_cluster_idx]

        matched_cluster = MatchedCluster(
            problem=profile["problem"],
            confidence=round(confidence, 2),
            occurrences=profile["occurrences"],
            solution_breakdown=profile.get("solution_breakdown", {}),
            typical_solution=profile.get("typical_solution", ""),
            average_manhours=profile.get("average_manhours", 0),
            parts_commonly_involved=profile.get("parts_commonly_involved", []),
        )

    # 5. Cross-reference parts against known high-failure parts
    related_parts: list[RelatedPart] = []
    for pn in all_part_numbers:
        if pn in index.part_by_number:
            p = index.part_by_number[pn]
            related_parts.append(RelatedPart(
                part_number=p["part_number"],
                failure_count=p["failure_count"],
                ai_summary=p.get("ai_summary", ""),
            ))

    # Also include parts from the matched cluster that are in part_failures
    # NOTE: Disabled — cluster-level parts include noise from concurrent/unrelated
    # maintenance sharing the same JCN (e.g., life raft inspections during gearbox work).
    # Parts from the actual similar TARs' MAFs (all_part_numbers above) are more precise.

    # Sort by failure count descending
    related_parts.sort(key=lambda p: -p.failure_count)

    return SearchResponse(
        matched_cluster=matched_cluster,
        similar_tars=similar_tars,
        related_parts=related_parts[:10],
        ai_recommendation=None,
    )
