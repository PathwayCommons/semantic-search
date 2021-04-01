from typing import List, Optional

import faiss

from pydantic import BaseModel, Field, validator
from transformers import PreTrainedModel, PreTrainedTokenizer

from semantic_search.ncbi import uids_to_docs

UID = str

# See: https://fastapi.tiangolo.com/tutorial/body/ for more details on creating a Request Body.


class Document(BaseModel):
    uid: UID
    text: Optional[str] = None


class Search(BaseModel):
    query: Document
    documents: List[Document]
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

    class Config:
        schema_extra = {
            "example": {
                "query": {
                    "uid": "0",
                    "text": "It has recently been shown that Craf is essential for Kras G12D-induced NSCLC.",
                },
                "documents": [
                    {
                        "uid": "1",
                        "text": "Craf is essential for the onset of Kras-driven non-small cell lung cancer.",
                    },
                    {
                        "uid": "2",
                        "text": "Tumorigenesis is a multistage process that involves multiple cell types.",
                    },
                    {
                        "uid": "3",
                        "text": "Only concomitant ablation of ERK1 and ERK2 impairs tumor growth.",
                    },
                ],
                "top_k": 3,
            }
        }


class TopMatch(BaseModel):
    uid: UID
    score: float


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    index: faiss.Index = None

    class Config:
        arbitrary_types_allowed = True
