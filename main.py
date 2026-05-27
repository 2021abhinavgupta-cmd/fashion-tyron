"""
Fashion Try-On Backend
======================
FastAPI server that handles the 3-image workflow:
  1. Product image (flat/standalone garment)
  2. Color reference (hex picked from this)
  3. Model image (garment placed on this model in the picked color)

Uses Kling AI kolors-virtual-try-on API.

Setup:
  pip install fastapi uvicorn python-multipart httpx requests pillow python-jose

Run:
  uvicorn main:app --reload --port 8000
"""

import os
import time
import hmac
import hashlib
import base64
import httpx
import json
import asyncio

from io import BytesIO
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ─── Config ────────────────────────────────────────────────────────────────────
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY", "YOUR_KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY", "YOUR_KLING_SECRET_KEY")
KLING_BASE_URL   = "https://api.klingai.com"

app = FastAPI(title="Fashion Try-On API", version="1.0.0")

# Allow requests from Google Flow / localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Kling JWT Auth ─────────────────────────────────────────────────────────────
def _build_kling_jwt() -> str:
    """Build a short-lived JWT for Kling AI API auth."""
    header  = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps({
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,   # 30 min
        "nbf": int(time.time()) - 5,
    }).encode()).rstrip(b"=")

    signing_input = header + b"." + payload
    sig = base64.urlsafe_b64encode(
        hmac.new(KLING_SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")

    return (signing_input + b"." + sig).decode()


def kling_headers() -> dict:
    return {
        "Authorization": f"Bearer {_build_kling_jwt()}",
        "Content-Type": "application/json",
    }


# ─── Color Swap Utility ─────────────────────────────────────────────────────────
def recolor_image_base64(base64_str: str, target_hex: str, tolerance: int = 45) -> str:
    """
    Pure-Python HSL hue-shift on the dominant garment color.
    Returns a new base64 PNG string.
    """
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")

    # Parse target hex → RGB → HSL
    th = target_hex.lstrip("#")
    tr, tg, tb = int(th[0:2], 16), int(th[2:4], 16), int(th[4:6], 16)
    target_hsl = _rgb_to_hsl(tr, tg, tb)

    pixels = img.load()
    w, h = img.size

    # Find dominant hue from center crop
    dominant_hue = _dominant_hue(pixels, w, h)

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a < 10:
                continue
            hsl = _rgb_to_hsl(r, g, b)
            diff = min(abs(hsl[0] - dominant_hue), 360 - abs(hsl[0] - dominant_hue))
            if diff < tolerance and hsl[1] > 8 and 5 < hsl[2] < 95:
                new_s = min(100, hsl[1] * (target_hsl[1] / max(1, hsl[1])) * 0.8 + hsl[1] * 0.2)
                nr, ng, nb = _hsl_to_rgb(target_hsl[0], new_s, hsl[2])
                pixels[x, y] = (nr, ng, nb, a)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _dominant_hue(pixels, w: int, h: int) -> float:
    counts: dict[int, int] = {}
    step = max(1, min(w, h) // 20)
    for y in range(int(h * 0.2), int(h * 0.8), step):
        for x in range(int(w * 0.2), int(w * 0.8), step):
            r, g, b = pixels[x, y][:3]
            hsl = _rgb_to_hsl(r, g, b)
            if hsl[1] > 15 and 10 < hsl[2] < 90:
                bucket = round(hsl[0] / 10) * 10
                counts[bucket] = counts.get(bucket, 0) + 1
    return max(counts, key=counts.get, default=0) if counts else 0


def _rgb_to_hsl(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return (0, 0, l * 100)
    d = mx - mn
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:   h = ((g - b) / d + (6 if g < b else 0)) / 6
    elif mx == g: h = ((b - r) / d + 2) / 6
    else:         h = ((r - g) / d + 4) / 6
    return (h * 360, s * 100, l * 100)


def _hsl_to_rgb(h, s, l):
    s /= 100; l /= 100
    k  = lambda n: (n + h / 30) % 12
    a  = s * min(l, 1 - l)
    f  = lambda n: l - a * max(-1, min(k(n) - 3, min(9 - k(n), 1)))
    return (round(f(0) * 255), round(f(8) * 255), round(f(4) * 255))


# ─── Kling Virtual Try-On ───────────────────────────────────────────────────────
async def _submit_tryon(human_b64: str, garment_b64: str) -> str:
    """Submit try-on task, returns task_id."""
    payload = {
        "model_name": "kolors-virtual-try-on-v1-5",   # latest as of 2025
        "human_image":   human_b64,
        "cloth_image":   garment_b64,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{KLING_BASE_URL}/v1/images/kolors-virtual-try-on",
            headers=kling_headers(),
            json=payload,
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Kling submit error {r.status_code}: {r.text}")
    data = r.json()
    if data.get("code") != 0:
        raise HTTPException(502, f"Kling API error: {data.get('message')}")
    return data["data"]["task_id"]


async def _poll_tryon(task_id: str, max_wait: int = 120) -> str:
    """Poll until done, returns result base64 image."""
    deadline = time.time() + max_wait
    async with httpx.AsyncClient(timeout=15) as client:
        while time.time() < deadline:
            r = await client.get(
                f"{KLING_BASE_URL}/v1/images/kolors-virtual-try-on/{task_id}",
                headers=kling_headers(),
            )
            if r.status_code != 200:
                raise HTTPException(502, f"Kling poll error: {r.text}")
            data = r.json()
            status = data["data"]["task_status"]

            if status == "succeed":
                url = data["data"]["task_result"]["images"][0]["url"]
                # Fetch the image and return as base64
                img_r = await client.get(url)
                return base64.b64encode(img_r.content).decode()

            elif status == "failed":
                raise HTTPException(502, f"Kling task failed: {data['data'].get('task_status_msg')}")

            await asyncio.sleep(3)

    raise HTTPException(504, "Try-on timed out after 120s")


# ─── Request / Response Models ──────────────────────────────────────────────────
class TryOnRequest(BaseModel):
    product_image_b64:  str          # Image 1 — flat product
    color_ref_b64:      str          # Image 2 — color reference
    model_image_b64:    str          # Image 3 — model photo
    target_hex:         str          # picked hex e.g. "#60614F"
    tolerance:          int  = 45    # color-swap tolerance
    mime_type:          str  = "image/png"


class TryOnResponse(BaseModel):
    result_b64:   str
    mime_type:    str = "image/png"
    task_id:      str


# ─── Main Endpoint ───────────────────────────────────────────────────────────────
@app.post("/try-on", response_model=TryOnResponse)
async def virtual_try_on(req: TryOnRequest):
    """
    3-image fashion try-on pipeline:
      1. Recolor the product image using target_hex
      2. Run Kling virtual try-on: place recolored product on model
      3. Return result as base64
    """

    # Step 1 — Recolor the product image
    try:
        recolored_b64 = recolor_image_base64(req.product_image_b64, req.target_hex, req.tolerance)
    except Exception as e:
        raise HTTPException(400, f"Color swap failed: {e}")

    # Step 2 — Submit to Kling virtual try-on
    task_id = await _submit_tryon(
        human_b64   = req.model_image_b64,
        garment_b64 = recolored_b64,
    )

    # Step 3 — Poll for result
    result_b64 = await _poll_tryon(task_id)

    return TryOnResponse(result_b64=result_b64, task_id=task_id)


@app.get("/health")
def health():
    return {"status": "ok", "service": "fashion-tryon-backend"}
