from io import BytesIO
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps
from pydantic import BaseModel, Field


SERVICE_ORDER = {"light": 0, "medium": 1, "heavy": 2}

TRUCK_MAP: Dict[str, Tuple[str, int, int]] = {
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

VOLUME_MAP: Dict[str, float] = {
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


class Context(BaseModel):
    place_type: Optional[str] = None
    place_size: Optional[str] = None
    pickup_floor: Optional[int] = None
    destination_floor: Optional[int] = None
    has_elevator: Optional[int] = None
    add_heavy: Optional[int] = None
    service_type: Optional[str] = None


class AnalyzeRequest(BaseModel):
    mode: str = "images"
    video_url: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    context: Context = Field(default_factory=Context)


app = FastAPI(title="MeezGo Direct AI Backend", version="1.0.0")
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def min_service_for_place(place_size: str) -> str:
    ls = (place_size or "").lower()
    min_service = "light"
    if ls.startswith("s3_") or ls.startswith("s4_"):
        min_service = "medium"
    if ls.startswith("villa_250_350"):
        min_service = "medium"
    if ls.startswith("villa_350_500") or ls.startswith("comm_150_250"):
        min_service = "heavy"
    if ls.startswith("comm_100_150"):
        min_service = "medium"
    return min_service


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_service_at_least(service_name: str, minimum: str) -> str:
    if SERVICE_ORDER.get(service_name, 0) < SERVICE_ORDER.get(minimum, 0):
        return minimum
    return service_name


def average_hash(gray_image: Image.Image, size: int = 8) -> np.ndarray:
    small = gray_image.resize((size, size), RESAMPLE_BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    return (arr >= arr.mean()).astype(np.uint8)


def hash_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def grayscale_metrics(gray_image: Image.Image) -> Dict[str, float]:
    arr = np.asarray(gray_image, dtype=np.float32)
    if arr.size == 0:
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "edge_density": 0.0,
            "blur_score": 0.0,
            "entropy": 0.0,
        }

    brightness = float(arr.mean() / 255.0)
    contrast = float(arr.std() / 255.0)

    dx = np.abs(arr[:, 1:] - arr[:, :-1])
    dy = np.abs(arr[1:, :] - arr[:-1, :])
    edge_mean = 0.0
    edge_density = 0.0
    if dx.size and dy.size:
        edge_strength = np.concatenate((dx.ravel(), dy.ravel()))
        edge_mean = float(edge_strength.mean() / 255.0)
        edge_density = float((edge_strength > 18.0).mean())

    blur_score = edge_mean

    hist, _ = np.histogram(arr, bins=32, range=(0, 255), density=False)
    total = hist.sum()
    entropy = 0.0
    if total > 0:
        probs = hist.astype(np.float64) / float(total)
        probs = probs[probs > 0]
        entropy = float(-(probs * np.log2(probs)).sum() / 5.0)

    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "edge_density": round(edge_density, 4),
        "blur_score": round(blur_score, 4),
        "entropy": round(entropy, 4),
    }


def download_image(url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content = BytesIO(response.content)
        image = Image.open(content)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception:
        return None


def analyze_image_set(image_urls: List[str]) -> Dict[str, Any]:
    inspected: List[Dict[str, Any]] = []
    hashes: List[np.ndarray] = []
    download_failures = 0
    duplicates_pruned = 0
    blurry_count = 0

    for raw_url in image_urls[:8]:
        image = download_image(raw_url)
        if image is None:
            download_failures += 1
            continue

        gray = ImageOps.grayscale(image)
        metrics = grayscale_metrics(gray)
        ahash = average_hash(gray)

        is_duplicate = any(hash_distance(ahash, existing) <= 5 for existing in hashes)
        if is_duplicate:
            duplicates_pruned += 1
            continue

        hashes.append(ahash)
        if metrics["blur_score"] < 0.04:
            blurry_count += 1

        inspected.append(
            {
                "width": int(image.width),
                "height": int(image.height),
                "metrics": metrics,
            }
        )

    if not inspected:
        return {
            "count": 0,
            "download_failures": download_failures,
            "duplicates_pruned": duplicates_pruned,
            "blurry_count": blurry_count,
            "avg_brightness": 0.0,
            "avg_contrast": 0.0,
            "avg_edge_density": 0.0,
            "avg_blur_score": 0.0,
            "avg_entropy": 0.0,
            "portrait_ratio": 0.0,
        }

    brightness_vals = [entry["metrics"]["brightness"] for entry in inspected]
    contrast_vals = [entry["metrics"]["contrast"] for entry in inspected]
    edge_vals = [entry["metrics"]["edge_density"] for entry in inspected]
    blur_vals = [entry["metrics"]["blur_score"] for entry in inspected]
    entropy_vals = [entry["metrics"]["entropy"] for entry in inspected]
    portrait_ratio = sum(1 for entry in inspected if entry["height"] >= entry["width"]) / float(len(inspected))

    return {
        "count": len(inspected),
        "download_failures": download_failures,
        "duplicates_pruned": duplicates_pruned,
        "blurry_count": blurry_count,
        "avg_brightness": round(float(np.mean(brightness_vals)), 4),
        "avg_contrast": round(float(np.mean(contrast_vals)), 4),
        "avg_edge_density": round(float(np.mean(edge_vals)), 4),
        "avg_blur_score": round(float(np.mean(blur_vals)), 4),
        "avg_entropy": round(float(np.mean(entropy_vals)), 4),
        "portrait_ratio": round(float(portrait_ratio), 4),
    }


def derive_visual_adjustment(context: Context, media: Dict[str, Any]) -> Dict[str, Any]:
    count = int(media.get("count", 0))
    contrast = float(media.get("avg_contrast", 0.0))
    edges = float(media.get("avg_edge_density", 0.0))
    entropy = float(media.get("avg_entropy", 0.0))
    blur_score = float(media.get("avg_blur_score", 0.0))
    portrait_ratio = float(media.get("portrait_ratio", 0.0))

    density_score = 0.0
    density_score += clamp((count - 2) / 6.0, 0.0, 0.35)
    density_score += clamp((contrast - 0.12) * 1.6, 0.0, 0.22)
    density_score += clamp((edges - 0.12) * 1.4, 0.0, 0.22)
    density_score += clamp((entropy - 0.55) * 0.35, 0.0, 0.16)
    if portrait_ratio > 0.65:
        density_score += 0.05
    if blur_score < 0.05:
        density_score -= 0.08

    density_score = clamp(density_score, -0.12, 0.72)

    volume_multiplier = 1.0 + density_score * 0.42
    if bool(context.add_heavy):
        volume_multiplier += 0.08

    complexity_points = 0
    if count >= 5:
        complexity_points += 1
    if count >= 7:
        complexity_points += 1
    if edges >= 0.18:
        complexity_points += 1
    if contrast >= 0.17:
        complexity_points += 1
    if bool(context.add_heavy):
        complexity_points += 1

    packing_complexity = "low"
    if complexity_points >= 4:
        packing_complexity = "high"
    elif complexity_points >= 2:
        packing_complexity = "medium"

    confidence = 0.54
    if count >= 3:
        confidence += 0.1
    if count >= 5:
        confidence += 0.08
    if media.get("duplicates_pruned", 0) > 0:
        confidence -= 0.03
    if media.get("download_failures", 0) > 0:
        confidence -= 0.06
    if media.get("blurry_count", 0) >= max(1, math.ceil(count * 0.6)):
        confidence -= 0.1
    if blur_score >= 0.05:
        confidence += 0.04

    confidence = clamp(confidence, 0.35, 0.9)

    return {
        "density_score": round(density_score, 4),
        "volume_multiplier": round(clamp(volume_multiplier, 0.9, 1.32), 4),
        "packing_complexity": packing_complexity,
        "quick_confidence": round(confidence, 4),
    }


def choose_service(context: Context, media: Dict[str, Any], visual: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    minimum = min_service_for_place(context.place_size or "")
    level = 1

    floors_total = int(context.pickup_floor or 0) + int(context.destination_floor or 0)
    if bool(context.add_heavy):
        level = max(level, 3)
        reasons.append("heavy_context_signal")
    if not bool(context.has_elevator) and floors_total >= 3:
        level = max(level, 2)
        reasons.append("stairs_without_elevator")

    count = int(media.get("count", 0))
    density_score = float(visual.get("density_score", 0.0))
    if count >= 6 or density_score >= 0.32:
        level = max(level, 2)
        reasons.append("visual_density_medium")
    if count >= 8 or density_score >= 0.5:
        level = max(level, 3)
        reasons.append("visual_density_high")

    service_name = "light"
    if level >= 3:
        service_name = "heavy"
    elif level >= 2:
        service_name = "medium"

    adjusted = ensure_service_at_least(service_name, minimum)
    if adjusted != service_name:
        reasons.append("place_size_minimum")
    service_name = adjusted

    return service_name, reasons


def build_inventory(req: AnalyzeRequest, media: Dict[str, Any], visual: Dict[str, Any]) -> Dict[str, Any]:
    ctx = req.context
    place_size = (ctx.place_size or "").lower()
    base_volume = VOLUME_MAP.get(place_size, 22.0)
    volume = round(base_volume * float(visual.get("volume_multiplier", 1.0)), 1)

    weight_low = round(volume * (11.5 + (2.0 if bool(ctx.add_heavy) else 0.0)), 1)
    weight_high = round(volume * (17.5 + (3.5 if bool(ctx.add_heavy) else 0.0)), 1)

    bulky_items: List[str] = []
    if bool(ctx.add_heavy):
        bulky_items.append("heavy_items_context")
    if volume >= 60:
        bulky_items.append("high_volume_load")

    fragile_items: List[str] = []
    if int(media.get("count", 0)) >= 5 and float(media.get("avg_contrast", 0.0)) >= 0.16:
        fragile_items.append("mixed_fragile_items_possible")

    access_signals: List[str] = []
    floors_total = int(ctx.pickup_floor or 0) + int(ctx.destination_floor or 0)
    if floors_total > 0 and not bool(ctx.has_elevator):
        access_signals.append("stairs_without_elevator")
    if floors_total >= 3:
        access_signals.append("multi_floor_move")

    disassembly_flags: List[str] = []
    if place_size.startswith("villa_") or place_size.startswith("s4_") or volume >= 70:
        disassembly_flags.append("large_furniture_likely")

    rooms_detected: List[str] = []
    if int(media.get("count", 0)) >= 2:
        rooms_detected.append("interior_space")
    if place_size.startswith("comm_"):
        rooms_detected.append("commercial_space")
    elif place_size.startswith("villa_"):
        rooms_detected.append("villa_home")
    elif place_size:
        rooms_detected.append("residential_unit")

    return {
        "rooms_detected": rooms_detected,
        "items_detected": [],
        "bulky_items": bulky_items,
        "fragile_items": fragile_items,
        "estimated_volume_m3": volume,
        "estimated_weight_kg_range": [weight_low, weight_high],
        "packing_complexity": visual.get("packing_complexity", "low"),
        "disassembly_flags": disassembly_flags,
        "access_signals": access_signals,
    }


def build_recommendations(
    req: AnalyzeRequest,
    media: Dict[str, Any],
    visual: Dict[str, Any],
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = req.context
    place_size = (ctx.place_size or "").lower()
    base_truck, base_workers, base_minutes = TRUCK_MAP.get(place_size, ("small", 2, 120))

    service_name, reasons = choose_service(ctx, media, visual)
    volume = float(inventory.get("estimated_volume_m3") or 22.0)
    floors_total = int(ctx.pickup_floor or 0) + int(ctx.destination_floor or 0)

    truck_name = base_truck
    truck_sequence = ["small", "medium", "large", "xl", "xxl"]
    truck_index = truck_sequence.index(truck_name) if truck_name in truck_sequence else 0
    if volume >= 55 and truck_index < len(truck_sequence) - 1:
        truck_index += 1
        reasons.append("volume_truck_upgrade")
    if bool(ctx.add_heavy) and truck_index < len(truck_sequence) - 1:
        truck_index += 1
        reasons.append("heavy_context_upgrade")
    truck_name = truck_sequence[min(truck_index, len(truck_sequence) - 1)]

    workers = base_workers
    if service_name == "medium":
        workers = max(workers, 3)
    if service_name == "heavy":
        workers = max(workers, 4)
    if volume >= 80:
        workers = max(workers, 5)
    if floors_total >= 4 and not bool(ctx.has_elevator):
        workers += 1
        reasons.append("extra_labor_for_stairs")

    estimated_minutes = base_minutes
    estimated_minutes = int(round(estimated_minutes * (0.95 + float(visual.get("volume_multiplier", 1.0)) * 0.32)))
    if floors_total > 0 and not bool(ctx.has_elevator):
        estimated_minutes += floors_total * 20
    if int(media.get("count", 0)) >= 6:
        estimated_minutes += 20
    if bool(ctx.add_heavy):
        estimated_minutes += 25

    confidence = float(visual.get("quick_confidence", 0.6))
    recommended_services = [service_name]
    if service_name == "light":
        recommended_services.append("medium")
    elif service_name == "medium":
        recommended_services.append("heavy")

    if not reasons:
        reasons.append("context_baseline_estimate")

    return {
        "service_type": service_name,
        "truck_size": truck_name,
        "workers": workers,
        "estimated_minutes": estimated_minutes,
        "load_class": service_name,
        "reasons": reasons,
        "confidence": round(confidence, 2),
        "recommended_services": recommended_services,
    }


def build_confidence(media: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, float]:
    base = float(recommendations.get("confidence", 0.6))
    count = float(media.get("count", 0))
    access = clamp(0.6 + (0.08 if count >= 3 else 0.0), 0.45, 0.86)
    volume = clamp(base + (0.04 if count >= 4 else -0.03), 0.35, 0.9)
    weight = clamp(volume - 0.07, 0.3, 0.85)
    service_type = clamp(base, 0.35, 0.9)
    return {
        "volume": round(volume, 2),
        "weight": round(weight, 2),
        "access": round(access, 2),
        "service_type": round(service_type, 2),
    }


def build_review_flags(req: AnalyzeRequest, media: Dict[str, Any], confidence: Dict[str, float]) -> List[str]:
    flags: List[str] = []
    if int(media.get("download_failures", 0)) > 0:
        flags.append("image_download_partial_failure")
    if int(media.get("duplicates_pruned", 0)) > 0:
        flags.append("duplicate_frames_pruned")
    if int(media.get("blurry_count", 0)) >= 2:
        flags.append("blurry_media_detected")
    if req.video_url and not req.image_urls:
        flags.append("video_without_frames")
    if float(confidence.get("volume", 0.0)) < 0.55:
        flags.append("low_visual_confidence")
    return flags


def manual_review_needed(req: AnalyzeRequest, media: Dict[str, Any], confidence: Dict[str, float]) -> bool:
    if req.video_url and not req.image_urls:
        return True
    if int(media.get("count", 0)) == 0:
        return True
    if float(confidence.get("volume", 0.0)) < 0.52:
        return True
    return False


def analyze_request(req: AnalyzeRequest) -> Dict[str, Any]:
    media = analyze_image_set(req.image_urls)
    visual = derive_visual_adjustment(req.context, media)
    inventory = build_inventory(req, media, visual)
    recommendations = build_recommendations(req, media, visual, inventory)
    confidence_by_field = build_confidence(media, recommendations)
    review_flags = build_review_flags(req, media, confidence_by_field)
    manual_review = manual_review_needed(req, media, confidence_by_field)

    telemetry = {
        "frame_count": len(req.image_urls),
        "fallback_reason": "" if media.get("count", 0) > 0 else ("video_only" if req.video_url else "heuristic_only"),
        "analysis_time_ms": 0,
        "usable_images": int(media.get("count", 0)),
        "download_failures": int(media.get("download_failures", 0)),
        "duplicates_pruned": int(media.get("duplicates_pruned", 0)),
    }

    analysis_media_mode = "mixed" if req.video_url and req.image_urls else ("video" if req.video_url else "images")

    return {
        "analysis_version": "v2-lite",
        "analysis_media_mode": analysis_media_mode,
        "source_type": req.mode or analysis_media_mode,
        "telemetry": telemetry,
        "inventory": inventory,
        "recommendations": recommendations,
        "confidence_by_field": confidence_by_field,
        "review_flags": review_flags,
        "manual_review_recommended": manual_review,
        "recommended_service": recommendations["service_type"],
        "truck_size": recommendations["truck_size"],
        "workers": recommendations["workers"],
        "estimated_minutes": recommendations["estimated_minutes"],
        "reasons": recommendations["reasons"],
        "confidence": recommendations["confidence"],
        "service_type_auto": recommendations["service_type"],
        "recommended_services": recommendations["recommended_services"],
        "quick_confidence": recommendations["confidence"],
        "needs_deep_pass": False,
    }


def verify_secret(request: Request) -> None:
    expected_secret = os.getenv("MCP_COLAB_API_SECRET", "").strip()
    if not expected_secret:
        return
    header_secret = request.headers.get("X-MCP-Secret", "").strip()
    if header_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid secret")


@app.get("/")
def root_health() -> Dict[str, Any]:
    return {"ok": True, "service": "meezgo-direct-ai-backend", "version": "1.0.0"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


def handle_analysis(req: AnalyzeRequest, request: Request) -> Dict[str, Any]:
    verify_secret(request)
    started = time.time()
    result = analyze_request(req)
    result["telemetry"]["analysis_time_ms"] = int((time.time() - started) * 1000)
    return result


@app.post("/")
async def analyze_root(req: AnalyzeRequest, request: Request) -> Dict[str, Any]:
    return handle_analysis(req, request)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, request: Request) -> Dict[str, Any]:
    return handle_analysis(req, request)
