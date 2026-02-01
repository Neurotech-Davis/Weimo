from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.utils.vlm import VLModel
from typing import Optional
from PIL import Image
import io

# Initialize router
vlm_router = APIRouter()

# Initialize model
vlm = VLModel()

# Define router
@vlm_router.post("/analyze")
async def analyze(
    system_prompt: Optional[str] = Form(None),
    message_prompt: Optional[str] = Form(None),
    image: UploadFile = File(...),
    max_new_tokens: Optional[int] = Form(None),
):
    try:
        # Validate image type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Form parameters
        system_prompt_value = system_prompt or ""
        message_prompt_value = message_prompt or ""

        # Inference
        result = vlm.inference(
            system_prompt=system_prompt_value, 
            message_prompt=message_prompt_value, 
            image=pil_image,
            max_new_tokens=max_new_tokens
        )
        
        return JSONResponse(content={"result": result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))