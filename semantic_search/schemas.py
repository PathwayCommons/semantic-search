from typing import List, Optional

import faiss
from pydantic import BaseModel, validator
from transformers import PreTrainedModel, PreTrainedTokenizer

from semantic_search.ncbi import uids_to_docs

UID = str

# See: https://fastapi.tiangolo.com/tutorial/body/ for more details on creating a Request Body.


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    index: faiss.Index = None

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    uid: UID
    text: str


class Query(BaseModel):
    query: Document
    documents: List[Document] = []
    top_k: Optional[int] = None

    @validator("query", "documents", pre=True)
    def normalize_document(cls, v, field):
        if field.name == "query":
            v = [v]

        normalized_docs = []
        for doc in v:
            if isinstance(doc, UID):
                normalized_docs.append(Document(**uids_to_docs([doc])[0]))
            else:
                normalized_docs.append(doc)
        return normalized_docs[0] if field.name == "query" else normalized_docs
