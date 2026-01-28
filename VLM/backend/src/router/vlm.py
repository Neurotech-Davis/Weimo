from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.vlm import (
    VLModel,
)
from pathlib import Path
from typing import Optional, Dict
import json

# Initialize router
vlm_router = APIRouter()

# Initialize model
vlm = VLModel()

# Define request schema
class URL(BaseModel):
    system_prompt: Optional[str] = None
    message_prompt: Optional[str] = None
    bs64_bytes: Optional[bytes] = None

# Define router
@vlm_router.post("/analyze")
def analyze(input: URL):

    # Define inputs
    system_prompt = input.system_prompt or ""
    message_prompt = input.message_prompt or ""
    bs64_bytes = input.bs64_bytes

    # Inference
    return vlm.inference(system_prompt=system_prompt, message_prompt=message_prompt, bs64_bytes=bs64_bytes)