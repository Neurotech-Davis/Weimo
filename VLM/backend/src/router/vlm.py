from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from VLM.backend.src.utils.vlm import (
    VLM,
)
from pathlib import Path
from typing import Optional, Dict
import json

# Initialize router
vlm_router = APIRouter()

# Initialize model
vlm = VLM()

# Define request schema
class URL(BaseModel):
    system_prompt: Optional[str] = None
    message_prompt: Optional[str] = None
    image_path: Optional[str]

# Define router
@vlm_router.post("/analyze")
def analyze(input: URL):

    # Define inputs
    system_prompt = input.system_prompt or ""
    message_prompt = input.message_prompt or ""
    image_path = input.image_path

    # Inference
    return vlm.inference(image_path=image_path)