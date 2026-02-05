from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os


class Context(BaseModel):
    place_type: Optional[str] = None
    place_size: Optional[str] = None
    pickup_floor: Optional[int] = None
    destination_floor: Optional[int] = None
    has_elevator: Optional[int] = None
    add_heavy: Optional[int] = None
    service_type: Optional[str] = None


class AnalyzeRequest(BaseModel):
    mode: str
    video_url: Optional[str] = None
    image_urls: List[str] = []
    context: Context


class Recommendations(BaseModel):
    service_type: str
    truck_size: Optional[str] = None
    workers: Optional[int] = None
    estimated_minutes: Optional[int] = None
    reasons: List[str] = []
    confidence: float = 0.7
    recommended_services: List[str] = []


class AnalyzeResponse(BaseModel):
    recommendations: Recommendations


app = FastAPI(title="MeezGo AI Media Analyzer", version="0.1.0")


def _compute_recommendations(req: AnalyzeRequest) -> Recommendations:
    ctx = req.context
    reasons: List[str] = []

    place_size = (ctx.place_size or "").lower()
    add_heavy = bool(ctx.add_heavy)
    floors_total = (ctx.pickup_floor or 0) + (ctx.destination_floor or 0)
    has_elevator = bool(ctx.has_elevator)

    # Minimum service from place size (approx mirror of plugin logic)
    min_service = "light"
    if place_size.startswith("s3_") or place_size.startswith("s4_"):
        min_service = "medium"
    if place_size.startswith("villa_250_350"):
        min_service = "medium"
    if place_size.startswith("villa_350_500") or place_size.startswith("comm_150_250"):
        min_service = "heavy"
    if place_size.startswith("comm_100_150"):
        min_service = "medium"

    suggest_level = 1
    if add_heavy:
        suggest_level = max(suggest_level, 3)
        reasons.append("vision_add_heavy_context")
    if not has_elevator and floors_total >= 3:
        suggest_level = max(suggest_level, 2)
        reasons.append("vision_multi_floor_no_elevator")

    # Very simple extra signal from number of images
    num_images = len(req.image_urls)
    if num_images >= 12:
        suggest_level = max(suggest_level, 3)
        reasons.append("vision_many_images_high_volume")
    elif num_images >= 6:
        suggest_level = max(suggest_level, 2)
        reasons.append("vision_medium_images_volume")

    svc = "light"
    if suggest_level >= 3:
        svc = "heavy"
    elif suggest_level >= 2:
        svc = "medium"

    order = {"light": 0, "medium": 1, "heavy": 2}
    if order.get(svc, 0) < order.get(min_service, 0):
        svc = min_service
        reasons.append("vision_place_size_minimum")

    truck_map = {
        "s1_40_60": ("small", 2, 90),
        "s1_60_80": ("small", 2, 120),
        "s2_80_100": ("medium", 3, 150),
        "s2_100_120": ("medium", 3, 180),
        "s3_120_150": ("large", 4, 240),
        "s3_150_180": ("large", 4, 300),
        "s4_180_220": ("xl", 4, 360),
        "s4_220_260": ("xl", 5, 420),
        "villa_250_350": ("xl", 4, 420),
        "villa_350_500": ("xxl", 5, 480),
        "comm_30_60": ("small", 2, 120),
        "comm_60_100": ("medium", 3, 180),
        "comm_100_150": ("large", 4, 240),
        "comm_150_250": ("xl", 5, 360),
    }
    t_size, workers, est_minutes = truck_map.get(place_size, ("small", 2, 120))

    if not has_elevator and floors_total > 0:
        est_minutes += floors_total * 20

    base_conf = 0.7
    if num_images >= 10:
        base_conf = 0.85
    elif num_images >= 5:
        base_conf = 0.78

    rec_services: List[str] = []
    if svc == "light":
        rec_services = ["light", "medium"]
    elif svc == "medium":
        rec_services = ["medium", "heavy"]
    else:
        rec_services = ["heavy"]

    return Recommendations(
        service_type=svc,
        truck_size=t_size,
        workers=workers,
        estimated_minutes=est_minutes,
        reasons=reasons,
        confidence=base_conf,
        recommended_services=rec_services,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_media(req: AnalyzeRequest, request: Request):
    expected_secret = os.getenv("MCP_COLAB_API_SECRET", "").strip()
    if expected_secret:
        header_secret = request.headers.get("X-MCP-Secret", "").strip()
        if header_secret != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid secret")

    rec = _compute_recommendations(req)
    return AnalyzeResponse(recommendations=rec)
