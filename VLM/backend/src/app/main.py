from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.config import Settings, get_settings
from src.router.vlm import vlm_router

# Setting Atttributes
# Attributes:
#     HF_KEY: str

app = FastAPI()

origins = [
    "http://localhost:3000",  # Front end (Not sure if I need this)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vlm_router, prefix="/api/v1/vlm")

@app.get("/ping/")
async def ping():
    return {"message": "pong"}

# TODO: REMOVE FOR DEPLOYMENT
@app.get("/env-check")  # ENV IS WORKING :)
async def env_check(settings: Annotated[Settings, Depends(get_settings)]):
    return settings
