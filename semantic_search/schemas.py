from typing import List

import faiss
from pydantic import BaseModel, validator, Field
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
    top_k: int = Field(10, gt=0, description="top_k must be greater than 0")

    @validator("query", "documents", pre=True)
    def normalize_document(cls, v, field):
        if field.name == "query":
            v = [v]

        normalized_docs = []
        for doc in v:
            if isinstance(doc, UID):
                # uids_to_docs expects a list of strings and yields a list of dictionaries. We
                # convert the generator to a list, and then index its first element, and then
                # unpack the dictionary and pass its contents as keyword arguments to Document.
                normalized_docs.append(Document(**list(uids_to_docs([doc]))[0][0]))
            else:
                normalized_docs.append(doc)
        return normalized_docs[0] if field.name == "query" else normalized_docs


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    index: faiss.Index = None

    class Config:
        arbitrary_types_allowed = True


class Response(BaseModel):
    uid: UID
    score: float
