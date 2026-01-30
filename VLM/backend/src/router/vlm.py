from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from src.utils.vlm import (
    VLModel,
)
from typing import Optional
from PIL import Image
import io

# Initialize router
vlm_router = APIRouter()

# Initialize model
vlm = VLModel()

# Define request schema
class Prompts(BaseModel):
    system_prompt: Optional[str] = None
    message_prompt: Optional[str] = None

# Define router
@vlm_router.post("/analyze")
async def analyze(system_prompt: Optional[str] = Form(None),
            message_prompt: Optional[str] = Form(None),
            image: UploadFile = File(...)):

    # Read image
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Define inputs
    system_prompt = input.system_prompt or ""
    message_prompt = input.message_prompt or ""
    image = pil_image

    # Inference
    return vlm.inference(system_prompt=system_prompt, message_prompt=message_prompt, image=image)