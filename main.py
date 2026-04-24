from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
import time


class Context(BaseModel):
    place_type: Optional[str] = None
    place_size: Optional[str] = None
    pickup_floor: Optional[int] = None
    destination_floor: Optional[int] = None
    has_elevator: Optional[int] = None
    add_heavy: Optional[int] = None
    service_type: Optional[str] = None


class AnalyzeMeta(BaseModel):
    analysis_media_mode: Optional[str] = "none"
    source_type: Optional[str] = "none"
    frame_count: int = 0


class AnalyzeRequest(BaseModel):
    mode: str
    video_url: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    context: Context
    meta: AnalyzeMeta = Field(default_factory=AnalyzeMeta)


class Inventory(BaseModel):
    rooms_detected: List[str] = Field(default_factory=list)
    items_detected: List[str] = Field(default_factory=list)
    bulky_items: List[str] = Field(default_factory=list)
    fragile_items: List[str] = Field(default_factory=list)
    estimated_volume_m3: Optional[float] = None
    estimated_weight_kg_range: Optional[List[float]] = None
    packing_complexity: Optional[str] = None
    disassembly_flags: List[str] = Field(default_factory=list)
    access_signals: List[str] = Field(default_factory=list)


class Recommendations(BaseModel):
    service_type: str
    truck_size: Optional[str] = None
    workers: Optional[int] = None
    estimated_minutes: Optional[int] = None
    load_class: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    recommended_services: List[str] = Field(default_factory=list)


class Telemetry(BaseModel):
    frame_count: int = 0
    fallback_reason: str = ""
    analysis_time_ms: int = 0


class AnalyzeResponse(BaseModel):
    analysis_version: str = "v2"
    analysis_media_mode: str = "none"
    source_type: str = "none"
    telemetry: Telemetry = Field(default_factory=Telemetry)
    inventory: Inventory = Field(default_factory=Inventory)
    recommendations: Recommendations
    confidence_by_field: Dict[str, float] = Field(default_factory=dict)
    review_flags: List[str] = Field(default_factory=list)
    manual_review_recommended: bool = False


app = FastAPI(title="MeezGo AI Media Analyzer", version="0.2.0")


def _compute_recommendations(req: AnalyzeRequest) -> Recommendations:
    ctx = req.context
    reasons: List[str] = []

    place_size = (ctx.place_size or "").lower()
    add_heavy = bool(ctx.add_heavy)
    floors_total = (ctx.pickup_floor or 0) + (ctx.destination_floor or 0)
    has_elevator = bool(ctx.has_elevator)

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
        load_class=svc,
        reasons=reasons,
        confidence=base_conf,
        recommended_services=rec_services,
    )


def _build_inventory(req: AnalyzeRequest) -> Inventory:
    ctx = req.context
    place_size = (ctx.place_size or "").lower()
    num_images = len(req.image_urls)

    volume_map = {
        "s1_40_60": 15.0,
        "s1_60_80": 20.0,
        "s2_80_100": 30.0,
        "s2_100_120": 35.0,
        "s3_120_150": 45.0,
        "s3_150_180": 55.0,
        "s4_180_220": 70.0,
        "s4_220_260": 85.0,
        "villa_250_350": 100.0,
        "villa_350_500": 140.0,
        "comm_30_60": 25.0,
        "comm_60_100": 40.0,
        "comm_100_150": 60.0,
        "comm_150_250": 90.0,
    }

    volume = volume_map.get(place_size)
    weight_range = None
    if volume is not None:
        low = round(volume * 12, 1)
        high = round(volume * 18, 1)
        weight_range = [low, high]

    bulky_items = []
    access_signals = []
    disassembly_flags = []

    if bool(ctx.add_heavy):
        bulky_items.append("heavy_items_context")

    floors_total = (ctx.pickup_floor or 0) + (ctx.destination_floor or 0)
    if not bool(ctx.has_elevator) and floors_total > 0:
        access_signals.append("stairs_without_elevator")

    if floors_total >= 3:
        access_signals.append("multi_floor_move")

    if place_size.startswith("villa_") or place_size.startswith("s4_"):
        disassembly_flags.append("large_furniture_likely")

    packing_complexity = None
    if num_images >= 10:
        packing_complexity = "medium"
    elif num_images >= 15:
        packing_complexity = "high"

    return Inventory(
        rooms_detected=[],
        items_detected=[],
        bulky_items=bulky_items,
        fragile_items=[],
        estimated_volume_m3=volume,
        estimated_weight_kg_range=weight_range,
        packing_complexity=packing_complexity,
        disassembly_flags=disassembly_flags,
        access_signals=access_signals,
    )


def _build_confidence_by_field(req: AnalyzeRequest, rec: Recommendations) -> Dict[str, float]:
    num_images = len(req.image_urls)

    volume_conf = 0.55
    if num_images >= 10:
        volume_conf = 0.85
    elif num_images >= 5:
        volume_conf = 0.72

    weight_conf = max(0.45, volume_conf - 0.12)
    access_conf = 0.65
    service_conf = rec.confidence

    return {
        "volume": round(volume_conf, 2),
        "weight": round(weight_conf, 2),
        "access": round(access_conf, 2),
        "service_type": round(service_conf, 2),
    }


def _build_review_flags(req: AnalyzeRequest, rec: Recommendations, inventory: Inventory) -> List[str]:
    flags: List[str] = []

    if req.mode == "video" and not req.image_urls:
        flags.append("video_fallback_mode")

    if rec.confidence < 0.75:
        flags.append("low_confidence_recommendation")

    if inventory.estimated_volume_m3 is None:
        flags.append("missing_volume_estimate")

    return flags


def _manual_review_needed(rec: Recommendations, review_flags: List[str]) -> bool:
    if rec.confidence < 0.65:
        return True
    if "missing_volume_estimate" in review_flags:
        return True
    return False


def _build_empty_response(req: AnalyzeRequest, fallback_reason: str, analysis_time_ms: int) -> AnalyzeResponse:
    default_rec = Recommendations(
        service_type=req.context.service_type or "medium",
        truck_size=None,
        workers=None,
        estimated_minutes=None,
        load_class=req.context.service_type or "medium",
        reasons=["no_media"],
        confidence=0.35,
        recommended_services=[],
    )

    return AnalyzeResponse(
        analysis_version="v2",
        analysis_media_mode=req.meta.analysis_media_mode or "none",
        source_type=req.meta.source_type or "none",
        telemetry=Telemetry(
            frame_count=req.meta.frame_count or 0,
            fallback_reason=fallback_reason,
            analysis_time_ms=analysis_time_ms,
        ),
        inventory=Inventory(),
        recommendations=default_rec,
        confidence_by_field={
            "volume": 0.2,
            "weight": 0.2,
            "access": 0.2,
            "service_type": 0.35,
        },
        review_flags=["no_media"],
        manual_review_recommended=True,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_media(req: AnalyzeRequest, request: Request):
    started = time.time()

    expected_secret = os.getenv("MCP_COLAB_API_SECRET", "").strip()
    if expected_secret:
        header_secret = request.headers.get("X-MCP-Secret", "").strip()
        if header_secret != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid secret")

    try:
        has_media = bool(req.image_urls) or bool(req.video_url)

        if not has_media:
            elapsed = int((time.time() - started) * 1000)
            return _build_empty_response(req, "no_media", elapsed)

        rec = _compute_recommendations(req)
        inventory = _build_inventory(req)
        confidence_by_field = _build_confidence_by_field(req, rec)
        review_flags = _build_review_flags(req, rec, inventory)
        manual_review = _manual_review_needed(rec, review_flags)

        elapsed = int((time.time() - started) * 1000)

        return AnalyzeResponse(
            analysis_version="v2",
            analysis_media_mode=req.meta.analysis_media_mode or ("images" if req.image_urls else "video"),
            source_type=req.meta.source_type or req.mode,
            telemetry=Telemetry(
                frame_count=req.meta.frame_count or 0,
                fallback_reason="",
                analysis_time_ms=elapsed,
            ),
            inventory=inventory,
            recommendations=rec,
            confidence_by_field=confidence_by_field,
            review_flags=review_flags,
            manual_review_recommended=manual_review,
        )

    except Exception:
        elapsed = int((time.time() - started) * 1000)
        return _build_empty_response(req, "exception", elapsed)
