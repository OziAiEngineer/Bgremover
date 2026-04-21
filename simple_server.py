"""
Simple IOPaint Server with minimal HTML UI
This bypasses the need to build the React frontend
"""
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# --- Static file directories ---
STATIC_DIR = Path(__file__).parent / "static"
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import IOPaint components
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, Device, HDStrategy, LDMSampler
from iopaint.helper import decode_base64_to_image, pil_to_bytes, concat_alpha_channel
from pydantic import BaseModel
from typing import Optional

class EraseRequest(BaseModel):
    """API request model — image & mask as base64 strings plus optional settings."""
    image: str                          # base64-encoded source image
    mask: str                           # base64-encoded B&W mask (white = erase)
    ldmSteps: int = 25
    hdStrategy: str = "Original"        # Original | Crop | Resize
    hdStrategyCropMargin: int = 128
    hdStrategyCropTrigerSize: int = 800
    hdStrategyResizeLimit: int = 2048

app = FastAPI(title="IOPaint Simple Eraser")

# Mount static files so images are accessible via URL
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler to safely return validation errors without crashing on binary data."""
    errors = []
    for err in exc.errors():
        # Safely stringify each error detail, avoiding binary data encoding issues
        safe_err = {
            "loc": list(err.get("loc", [])),
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        errors.append(safe_err)
    return JSONResponse(
        status_code=422,
        content={
            "detail": errors,
            "hint": "For /api/v1/inpaint send JSON with base64 'image' and 'mask' fields. "
                    "To upload raw files use /api/v1/inpaint-file with multipart/form-data."
        }
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
print("Initializing LaMa model...")
model_manager = ModelManager(
    name="lama",
    device=torch.device("cpu"),
    no_half=False,
    low_mem=False,
    disable_nsfw=False,
    sd_cpu_textencoder=False,
    local_files_only=False,
    cpu_offload=False,
)
print("LaMa model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the simple HTML UI"""
    html_file = Path(__file__).parent / "simple_eraser.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'), status_code=200)
    return HTMLResponse(content="<h1>Error: simple_eraser.html not found</h1>", status_code=404)

def _save_webp(pil_image: Image.Image, directory: Path) -> str:
    """Save a PIL image as WebP and return its filename."""
    filename = f"{uuid.uuid4().hex}.webp"
    pil_image.save(directory / filename, format="WEBP", quality=90)
    return filename


def _make_url(request: Request, path: str) -> str:
    """Build an absolute URL for a static file path."""
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"


@app.post("/api/v1/inpaint")
async def inpaint(req: EraseRequest, request: Request):
    """Process inpainting request — saves result as WebP, returns URL.
    
    Body (JSON):
      image  : base64 string of the source image
      mask   : base64 string of a B&W PNG (white = erase, black = keep)
    """
    try:
        # Decode base64 images
        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)

        # Threshold mask
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Check dimensions match
        if image.shape[:2] != mask.shape[:2]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Image size {image.shape[:2]} and mask size {mask.shape[:2]} don't match"},
            )

        # Process with model (build InpaintRequest from settings)
        inpaint_req = InpaintRequest(
            ldmSteps=req.ldmSteps,
            hdStrategy=req.hdStrategy,
            hdStrategyCropMargin=req.hdStrategyCropMargin,
            hdStrategyCropTrigerSize=req.hdStrategyCropTrigerSize,
            hdStrategyResizeLimit=req.hdStrategyResizeLimit,
        )
        print(f"Processing image of size {image.shape[:2]}...")
        rgb_np_img = model_manager(image, mask, inpaint_req)
        print("Processing complete!")

        # Convert result back to RGB and save as WebP
        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
        result_filename = _save_webp(Image.fromarray(rgb_res), RESULTS_DIR)
        result_url = _make_url(request, f"/static/results/{result_filename}")

        return JSONResponse(content={"result_url": result_url})

    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/v1/inpaint-file")
async def inpaint_file(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    request: Request = None,
):
    """Process inpainting from multipart/form-data — saves result as WebP, returns URL."""
    try:
        from iopaint.helper import load_img

        # Read files into bytes
        image_bytes = await image.read()
        mask_bytes = await mask.read()

        # Load using IOPaint helper
        np_image, alpha_channel, infos = load_img(image_bytes, return_info=True)
        np_mask, _ = load_img(mask_bytes, gray=True)

        # Threshold mask
        np_mask = cv2.threshold(np_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Check dimensions match
        if np_image.shape[:2] != np_mask.shape[:2]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Image size {np_image.shape[:2]} and mask size {np_mask.shape[:2]} don't match"},
            )

        # Process with model
        print(f"Processing image of size {np_image.shape[:2]}...")
        inpaint_req = InpaintRequest()
        rgb_np_img = model_manager(np_image, np_mask, inpaint_req)
        print("Processing complete!")

        # Convert result back to RGB and save as WebP
        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
        result_filename = _save_webp(Image.fromarray(rgb_res), RESULTS_DIR)
        result_url = _make_url(request, f"/static/results/{result_filename}")

        return JSONResponse(content={"result_url": result_url})

    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model": "lama"}

if __name__ == "__main__":
    HOST = "172.16.2.231"
    PORT = 8080
    
    print("\n" + "="*60)
    print("🎨 IOPaint Simple Eraser Server")
    print("="*60)
    print(f"\n📍 Server starting at: http://{HOST}:{PORT}")
    print(f"🌐 Open your browser and go to: http://{HOST}:{PORT}")
    print("\n✨ Ready to erase objects from images!")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
