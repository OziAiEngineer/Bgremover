"""
Simple IOPaint Server with minimal HTML UI
This bypasses the need to build the React frontend
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Import IOPaint components
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, Device
from iopaint.helper import decode_base64_to_image, pil_to_bytes, concat_alpha_channel

app = FastAPI(title="IOPaint Simple Eraser")

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

@app.post("/api/v1/inpaint")
async def inpaint(req: InpaintRequest):
    """Process inpainting request"""
    try:
        # Decode base64 images
        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        
        # Threshold mask
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Check dimensions match
        if image.shape[:2] != mask.shape[:2]:
            return Response(
                content=f"Image size {image.shape[:2]} and mask size {mask.shape[:2]} don't match",
                status_code=400
            )
        
        # Process with model
        print(f"Processing image of size {image.shape[:2]}...")
        rgb_np_img = model_manager(image, mask, req)
        print("Processing complete!")
        
        # Convert back to RGB
        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
        
        # Convert to bytes
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=95,
            infos=infos,
        )
        
        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
        )
    
    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Error: {str(e)}",
            status_code=500
        )

@app.post("/api/v1/inpaint-file")
async def inpaint_file(
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    """Process inpainting request from multipart/form-data file uploads"""
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
            return Response(
                content=f"Image size {np_image.shape[:2]} and mask size {np_mask.shape[:2]} don't match",
                status_code=400
            )
        
        # Process with model
        print(f"Processing image of size {np_image.shape[:2]}...")
        req = InpaintRequest()
        rgb_np_img = model_manager(np_image, np_mask, req)
        print("Processing complete!")
        
        # Convert back to RGB
        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)
        
        # Convert to bytes
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext="jpeg",
            quality=95,
            infos=infos,
        )
        
        return Response(
            content=res_img_bytes,
            media_type="image/jpeg",
        )
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Error: {str(e)}",
            status_code=500
        )

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
