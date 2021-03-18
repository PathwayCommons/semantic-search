from typing import List

import faiss
from pydantic import BaseModel, validator
from transformers import PreTrainedModel, PreTrainedTokenizer

from semantic_search.ncbi import uids_to_docs

UID = str

# See: https://fastapi.tiangolo.com/tutorial/body/ for more details on creating a Request Body.


class Document(BaseModel):
    uid: UID
    text: str


class Query(BaseModel):
    query: Document
    documents: List[Document] = []
    top_k: int = 10

    @validator("query", "documents", pre=True)
    def normalize_document(cls, v, field):
        if field.name == "query":
            v = [v]

        normalized_docs = []
        for doc in v:
            if isinstance(doc, UID):
                normalized_docs.append(Document(**list(uids_to_docs([doc]))[0][0]))
            else:
                normalized_docs.append(doc)
        return normalized_docs[0] if field.name == "query" else normalized_docs

    @validator("top_k")
    def top_k_must_be_gt_zero(cls, v):
        if not v > 0:
            raise ValueError(f"top_k must be greater than 0, got {v}")
        return v


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    index: faiss.Index = None

    class Config:
        arbitrary_types_allowed = True


class Response(BaseModel):
    uid: UID
    score: float
