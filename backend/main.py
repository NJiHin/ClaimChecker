import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import pipeline
from mlx_lm import load
from google import genai

load_dotenv()

from src.kb.database import init_db, prune_claim_cache
from src.api.routes import router

_CLASSIFIER_MODEL = "lighteternal/fact-or-opinion-xlmr-el"
_APS_MODEL = str((Path(__file__).parent.parent / "mlx_model").resolve())

_ready_event = asyncio.Event()


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


async def _load_models(app: FastAPI):
    device = _resolve_device()
    app.state.classifier = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pipeline("text-classification", model=_CLASSIFIER_MODEL, device=device),
    )
    model, tokeniser = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: load(_APS_MODEL),
    )
    app.state.aps = (model, tokeniser)
    _ready_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=2,
        max_size=10,
    )
    async with pool.acquire() as conn:
        await init_db(conn)
        await prune_claim_cache(conn)
    app.state.db = pool
    app.state.pipeline_states = {}
    app.state.gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    asyncio.create_task(_load_models(app))

    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/ready")
async def ready():
    await _ready_event.wait()
    return {"ready": True}