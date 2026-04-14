import asyncio
import json
from typing import AsyncGenerator

import asyncpg
from google import genai

from src.pipeline.decompose import classify, extract_propositions
from src.pipeline.verify import verify

def _sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def run_pipeline(
    sentences: list[str],
    classifier,   # transformers text-classification pipeline (app.state.classifier)
    aps,          # mlx model and tokeniser (app.state.aps)
    client: genai.Client,
    pool: asyncpg.Pool,
    stop_flag: list[bool],
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Async generator that runs classify → extract and yields SSE strings.

    stop_flag is a single-element list shared with the caller so the route can
    set stop_flag[0] = True to cancel between sub-batches.
    """

    yield _sse("session", session_id)

    # --- classify ---
    yield _sse("status", "classifying")

    facts = await classify(sentences, classifier, stop_flag=stop_flag)

    if stop_flag[0]:
        yield _sse("error", "cancelled")
        return

    # --- extract propositions ---
    yield _sse("status", "extracting_propositions")

    sentence_props = await extract_propositions(facts, aps, stop_flag=stop_flag)

    if stop_flag[0]:
        yield _sse("error", "cancelled")
        return

    # --- verify ---
    yield _sse("status", "verifying")

    props_list = []
    for sentence, props in sentence_props.items():
        for prop in props:
            props_list.append((sentence, prop))

    verdicts = await asyncio.gather(*[
        verify(prop, pool, client)
        for _, prop in props_list
    ])

    propositions = []
    for (sentence, prop), verdict in zip(props_list, verdicts):
        propositions.append({"sentence": sentence, "text": prop, "verdict": verdict})

    yield _sse("result", propositions)
