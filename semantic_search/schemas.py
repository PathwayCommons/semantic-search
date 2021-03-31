from typing import List, Optional

import faiss
from pydantic import BaseModel, Field
from transformers import PreTrainedModel, PreTrainedTokenizer

UID = str

# See: https://fastapi.tiangolo.com/tutorial/body/ for more details on creating a Request Body.


class Document(BaseModel):
    uid: UID
    text: Optional[str] = None


class Query(BaseModel):
    query: Document
    documents: List[Document] = []
    top_k: int = Field(10, gt=0, description="top_k must be greater than 0")


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    index: faiss.Index = None

    class Config:
        arbitrary_types_allowed = True


class Response(BaseModel):
    uid: UID
    score: float
