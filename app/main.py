"""FastAPI application — TAR Intelligence System."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.indexer import index, load_index
from app.llm import generate_recommendation
from app.models import (
    RecommendRequest, RecommendResponse, SearchRequest, SearchResponse,
    StatsResponse,
)
from app.search import search_tar
from app.tpdr import run_tpdr_analysis, get_tpdr_results, get_system_detail

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    run_tpdr_analysis()
    yield


app = FastAPI(
    title="V-22 TAR Intelligence System",
    description="AI-powered maintenance decision support for V-22 Osprey",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- API Endpoints ---

@app.post("/api/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")
    try:
        return search_tar(req.text, req.top_k)
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")


@app.post("/api/recommend", response_model=RecommendResponse)
async def api_recommend(req: RecommendRequest):
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")

    cluster = req.search_results.get("matched_cluster", {})
    cluster_label = cluster.get("problem", "Unknown")
    breakdown = cluster.get("solution_breakdown", {})
    avg_mh = cluster.get("average_manhours", 0)

    # Gather corrective actions from similar TARs
    corr_actions = []
    for tar in req.search_results.get("similar_tars", []):
        for maf in tar.get("maf_actions", []):
            ca = maf.get("corr_act", "").strip()
            if ca:
                corr_actions.append(ca)

    try:
        rec = generate_recommendation(
            req.text, cluster_label, breakdown, avg_mh, corr_actions
        )
        return RecommendResponse(recommendation=rec)
    except Exception as e:
        raise HTTPException(500, f"Recommendation generation failed: {e}")


@app.get("/api/clusters")
async def api_clusters():
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")
    return index.cluster_profiles


@app.get("/api/parts")
async def api_parts():
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")
    return index.part_failures


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    return StatsResponse(
        total_tars=len(index.tar_df) if index.loaded else 0,
        total_mafs=index.total_maf_records,
        cluster_count=len(index.cluster_profiles),
        parts_tracked=len(index.part_failures),
        index_loaded=index.loaded,
    )


# --- TAR Queue Endpoints ---

@app.get("/api/tars/recent")
async def api_tars_recent(
    limit: int = Query(50, ge=1, le=200),
    work_center: str = Query("", alias="work_center"),
    activity: str = Query("", alias="activity"),
    priority: str = Query("", alias="priority"),
    search: str = Query(""),
):
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")

    df = index.tar_df

    # Apply filters
    if work_center:
        df = df[df["work_center"].str.strip() == work_center]
    if activity:
        df = df[df["activity"].str.strip() == activity]
    if priority:
        df = df[df["priority"].str.strip() == priority]
    if search:
        search_lower = search.lower()
        combined = (df["subject"].str.strip() + " " + df["issue"].str.strip()).str.lower()
        df = df[combined.str.contains(search_lower, na=False)]

    # Sort by submit_dt descending, NaT last
    df = df.sort_values("submit_dt", ascending=False, na_position="last")

    # Limit
    df = df.head(limit)

    # Build response
    records = []
    for _, row in df.iterrows():
        records.append({
            "jcn": str(row.get("jcn", "")).strip(),
            "subject": str(row.get("subject", "")).strip(),
            "issue": str(row.get("issue", "")).strip(),
            "uns": str(row.get("uns", "")).strip(),
            "submit_date": str(row.get("submit_date", "")).strip(),
            "priority": str(row.get("priority", "")).strip(),
            "activity": str(row.get("activity", "")).strip(),
            "work_center": str(row.get("work_center", "")).strip(),
            "status": str(row.get("status", "")).strip(),
            "buno": str(row.get("buno", "")).strip(),
            "aircraft_type": str(row.get("aircraft_type", "")).strip(),
        })

    return records


@app.get("/api/tars/filters")
async def api_tars_filters():
    if not index.loaded:
        raise HTTPException(503, "Index not loaded yet")

    df = index.tar_df

    def unique_sorted(col):
        vals = df[col].str.strip().dropna().unique().tolist()
        return sorted([v for v in vals if v])

    return {
        "work_centers": unique_sorted("work_center"),
        "activities": unique_sorted("activity"),
        "priorities": unique_sorted("priority"),
    }


# --- TPDR Endpoints ---

@app.get("/api/tpdr/recommendations")
async def api_tpdr_recommendations():
    results = get_tpdr_results()
    if results is None:
        raise HTTPException(503, "TPDR analysis not yet complete")
    return {
        "recommendations": results.get("recommendations", []),
        "computed_at": results.get("computed_at", ""),
    }


@app.get("/api/tpdr/trends")
async def api_tpdr_trends():
    results = get_tpdr_results()
    if results is None:
        raise HTTPException(503, "TPDR analysis not yet complete")
    return {"trends": results.get("trends", [])}


@app.get("/api/tpdr/comebacks")
async def api_tpdr_comebacks():
    results = get_tpdr_results()
    if results is None:
        raise HTTPException(503, "TPDR analysis not yet complete")
    return {"comebacks": results.get("comebacks", [])}


@app.get("/api/tpdr/system/{uns_code:path}")
async def api_tpdr_system_detail(uns_code: str):
    detail = get_system_detail(uns_code)
    if detail is None:
        raise HTTPException(404, f"System '{uns_code}' not found")
    return {"system": detail}


@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))
