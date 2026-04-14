import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from src.kb.ingest import ingest_pdf
from src.models import AbortRequest, PipelineRequest
from src.pipeline.pipeline import run_pipeline

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency — pool connection
# ---------------------------------------------------------------------------

async def get_conn(request: Request):
    async with request.app.state.db.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# GET /kb/docs
# ---------------------------------------------------------------------------

@router.get("/kb/docs")
async def list_docs(conn=Depends(get_conn)):
    rows = await conn.fetch(
        """
        SELECT doc_id, doc_title, MAX(created_at) AS latest_version_at
        FROM passages
        GROUP BY doc_id, doc_title
        ORDER BY latest_version_at DESC
        """
    )
    return [
        {
            "doc_id": r["doc_id"],
            "doc_title": r["doc_title"],
            "latest_version_at": r["latest_version_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# POST /kb/upload
# ---------------------------------------------------------------------------

@router.post("/kb/upload")
async def upload_doc(file: UploadFile, conn=Depends(get_conn)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    count = await ingest_pdf(pdf_bytes, file.filename, conn)
    return {"passages_stored": count}


# ---------------------------------------------------------------------------
# DELETE /kb/delete
# ---------------------------------------------------------------------------

@router.delete("/kb/delete")
async def delete_doc(doc_id: str, conn=Depends(get_conn)):
    doc_title = await conn.fetchval(
        "SELECT doc_title FROM passages WHERE doc_id = $1 LIMIT 1",
        doc_id,
    )
    if doc_title is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    await conn.execute("DELETE FROM passages WHERE doc_id = $1", doc_id)
    return {"deleted_file": doc_title}


# ---------------------------------------------------------------------------
# POST /api/claims/pipeline/run  (SSE)
# ---------------------------------------------------------------------------

@router.post("/api/claims/pipeline/run")
async def pipeline_run(body: PipelineRequest, request: Request):
    session_id = secrets.token_hex(16)
    stop_flag = [False]
    request.app.state.pipeline_states[session_id] = stop_flag

    async def generate():
        try:
            async for chunk in run_pipeline(
                body.sentences,
                request.app.state.classifier,
                request.app.state.aps,
                request.app.state.gemini_client,
                request.app.state.db,
                stop_flag,
                session_id,
            ):
                yield chunk
        finally:
            request.app.state.pipeline_states.pop(session_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /api/claims/pipeline/abort
# ---------------------------------------------------------------------------

@router.post("/api/claims/pipeline/abort")
async def pipeline_abort(body: AbortRequest, request: Request):
    flag = request.app.state.pipeline_states.get(body.session_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    flag[0] = True
    return {"aborted": True}
