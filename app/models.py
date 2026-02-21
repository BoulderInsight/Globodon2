"""Pydantic request/response models for the TAR Intelligence API."""

from pydantic import BaseModel, Field


# --- Requests ---

class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, description="TAR description text to search")
    top_k: int = Field(10, ge=1, le=50, description="Number of similar TARs to return")


class RecommendRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Original TAR text")
    search_results: dict = Field(..., description="Results from /api/search")


# --- Response components ---

class MAFAction(BaseModel):
    corr_act: str = ""
    action_taken: str = ""
    manhours: str = ""
    inst_partno: str = ""
    rmvd_partno: str = ""


class SimilarTAR(BaseModel):
    jcn: str = ""
    subject: str = ""
    issue: str = ""
    similarity: float = 0.0
    cluster_label: str = ""
    maf_actions: list[MAFAction] = []


class MatchedCluster(BaseModel):
    problem: str = ""
    confidence: float = 0.0
    occurrences: int = 0
    solution_breakdown: dict[str, float] = {}
    typical_solution: str = ""
    average_manhours: float = 0.0
    parts_commonly_involved: list[str] = []


class RelatedPart(BaseModel):
    part_number: str = ""
    failure_count: int = 0
    ai_summary: str = ""


class SearchResponse(BaseModel):
    matched_cluster: MatchedCluster | None = None
    similar_tars: list[SimilarTAR] = []
    related_parts: list[RelatedPart] = []
    ai_recommendation: str | None = None


class RecommendResponse(BaseModel):
    recommendation: str = ""


class StatsResponse(BaseModel):
    total_tars: int = 0
    total_mafs: int = 0
    cluster_count: int = 0
    parts_tracked: int = 0
    index_loaded: bool = False
