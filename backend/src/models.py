from pydantic import BaseModel

class PipelineRequest(BaseModel):
    sentences: list[str]


class AbortRequest(BaseModel):
    session_id: str
