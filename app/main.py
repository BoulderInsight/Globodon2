"""FastAPI application — TAR Intelligence System."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.indexer import index, load_index
from app.llm import generate_recommendation
from app.models import (
    RecommendRequest, RecommendResponse, SearchRequest, SearchResponse,
    StatsResponse,
)
from app.search import search_tar

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
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


@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))
