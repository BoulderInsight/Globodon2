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
    buno: str = ""
    activity: str = ""
    submit_date: str = ""
    status: str = ""
    priority: str = ""


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
    recommendation: dict = {}


class StatsResponse(BaseModel):
    total_tars: int = 0
    total_mafs: int = 0
    cluster_count: int = 0
    parts_tracked: int = 0
    index_loaded: bool = False


# --- TPDR Models ---

class TPDRSampleTar(BaseModel):
    jcn: str = ""
    subject: str = ""
    submit_date: str = ""
    buno: str = ""


class TPDRComeback(BaseModel):
    buno: str = ""
    first_jcn: str = ""
    second_jcn: str = ""
    gap_days: float = 0
    date_first: str = ""
    date_second: str = ""


class TPDRCandidate(BaseModel):
    uns: str = ""
    score: float = 0
    total_tars: int = 0
    first_half_count: int = 0
    second_half_count: int = 0
    acceleration_ratio: float = 0
    monthly_counts: list[int] = []
    months_labels: list[str] = []
    affected_aircraft: list[str] = []
    aircraft_count: int = 0
    priority_breakdown: dict[str, int] = {}
    activities: list[str] = []
    urgent_priority_count: int = 0
    linked_cluster: str = ""
    sample_tars: list[TPDRSampleTar] = []
    justification: str = ""
    comeback_count: int = 0
    comeback_aircraft: int = 0
    avg_gap_days: float = 0
    common_fixes: dict[str, int] = {}
    example_comebacks: list[TPDRComeback] = []


class TPDRTrend(BaseModel):
    uns: str = ""
    total_tars: int = 0
    acceleration_ratio: float = 0
    monthly_counts: list[int] = []
    months_labels: list[str] = []
    aircraft_count: int = 0
    first_half_count: int = 0
    second_half_count: int = 0
    linked_cluster: str = ""


class TPDRComebackSummary(BaseModel):
    uns: str = ""
    comeback_count: int = 0
    unique_aircraft: int = 0
    avg_gap_days: float = 0
    common_fixes: dict[str, int] = {}
    example_comebacks: list[TPDRComeback] = []


class TPDRRecommendationsResponse(BaseModel):
    recommendations: list[TPDRCandidate] = []
    computed_at: str = ""


class TPDRTrendsResponse(BaseModel):
    trends: list[TPDRTrend] = []


class TPDRComebacksResponse(BaseModel):
    comebacks: list[TPDRComebackSummary] = []


class TPDRSystemDetailResponse(BaseModel):
    system: dict = {}


# --- UNS System View Models ---

class UNSCorrectiveAction(BaseModel):
    action: str = ""
    count: int = 0
    avg_manhours: float = 0.0


class UNSFailureMode(BaseModel):
    mode_id: int = 0
    count: int = 0
    label: str = ""
    sample_subjects: list[str] = []
    top_actions: list[str] = []
    silhouette_score: float = 0.0


class UNSPartCount(BaseModel):
    part_number: str = ""
    count: int = 0


class UNSSystem(BaseModel):
    uns_code: str = ""
    uns_name: str = ""
    total_tars: int = 0
    aircraft_affected: int = 0
    activities: list[str] = []
    date_range: dict[str, str] = {}
    monthly_counts: list[int] = []
    months_labels: list[str] = []
    top_corrective_actions: list[UNSCorrectiveAction] = []
    failure_modes: list[UNSFailureMode] = []
    sub_cluster_quality: float | None = None
    top_parts: list[UNSPartCount] = []


class UNSSystemsResponse(BaseModel):
    systems: list[UNSSystem] = []
    total_systems: int = 0
    computed_at: str = ""
