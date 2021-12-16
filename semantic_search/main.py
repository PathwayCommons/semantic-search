from datetime import datetime
from http import HTTPStatus
from operator import itemgetter
from typing import List, Optional, Tuple, Union, cast

import faiss
import torch
from fastapi import FastAPI, Request
from pydantic import BaseSettings

from semantic_search import __version__
from semantic_search.common.util import (
    add_to_faiss_index,
    encode_with_transformer,
    setup_faiss_index,
    setup_model_and_tokenizer,
    normalize_documents,
)
from semantic_search.schemas import Model, Search, TopMatch
from loguru import logger
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from fastapi import HTTPException

dot_env_filepath = Path(__file__).absolute().parent.parent / ".env"
load_dotenv(dot_env_filepath)

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {level} | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "DEBUG"),
)

app = FastAPI(
    title="Scientific Semantic Search",
    description="A simple semantic search engine for scientific papers.",
    version=__version__,
)


class Settings(BaseSettings):
    """Store global settings for the web-service. Pass these as environment variables at server
    startup. E.g.

    `CUDA_DEVICE=0 MAX_LENGTH=384 uvicorn semantic_search.main:app`
    """

    pretrained_model_name_or_path: str = "johngiorgi/declutr-sci-base"
    batch_size: int = 64
    max_length: Optional[int] = None
    mean_pool: bool = True
    cuda_device: int = -1


settings = Settings()
model = Model()


def encode(text: Union[str, List[str]]) -> torch.Tensor:
    if isinstance(text, str):
        text = [text]
    # Sort the inputs by length, maintaining the original indices so we can un-sort
    # before returning the embeddings. This speeds up embedding by minimizing the
    # amount of computation performed on pads. Because this sorting happens before
    # tokenization, it is only a proxy of the true lengths of the inputs to the model.
    # In the future, it would be better to sort by length *after* tokenization which
    # would lead to an even larger speedup.
    # https://stackoverflow.com/questions/8372399/zip-with-list-output-instead-of-tuple
    sorted_indices, text = cast(
        Tuple[Tuple[int], List[str]], zip(*sorted(enumerate(text), key=itemgetter(1)))
    )  # tell mypy explicitly the types of items in the unpacked tuple
    unsorted_indices, _ = zip(*sorted(enumerate(sorted_indices), key=itemgetter(1)))

    embeddings: torch.Tensor = []
    for i in range(0, len(text), settings.batch_size):
        embedding = encode_with_transformer(
            list(text[i : i + settings.batch_size]),
            tokenizer=model.tokenizer,
            model=model.model,
            mean_pool=settings.mean_pool,
        )
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings)

    # Unsort the embedded text so that it is returned in the same order it was recieved.
    unsorted_indices = torch.as_tensor(unsorted_indices, dtype=torch.long, device=embeddings.device)
    embeddings = torch.index_select(embeddings, dim=0, index=unsorted_indices)

    return embeddings


@app.on_event("startup")
def app_startup():

    model.tokenizer, model.model = setup_model_and_tokenizer(
        settings.pretrained_model_name_or_path, cuda_device=settings.cuda_device
    )
    embedding_dim = model.model.config.hidden_size
    model.index = setup_faiss_index(embedding_dim)


@app.middleware("http")
async def log_middle(request: Request, call_next):

    response = await call_next(request)
    status = response.status_code
    method = request.method
    path = request.url.path
    user_agent = request.headers.get("User-Agent")
    logger.info(f"{method} {path} {status} {user_agent}")

    return response


@app.get("/", tags=["General"])
def index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "method": request.method,
        "status-code": HTTPStatus.OK,
        "timestamp": datetime.now().isoformat(),
        "url": request.url._url,
    }
    return response


@app.post("/search", tags=["Search"], response_model=List[TopMatch])
async def search(search: Search):
    """Returns the `top_k` most similar documents to `query` from the provided list of `documents`
    and the index. When docs_only is True, returns all `documents` provided, and disregards `top_k`.
    """
    ids = [int(doc.uid) for doc in search.documents]
    texts = [document.text for document in search.documents]

    # Only add items to the index if they do not already exist.
    # See: https://github.com/facebookresearch/faiss/issues/859
    # To do this, we first determine which of the incoming ids do not exist in the index
    indexed_ids = set(faiss.vector_to_array(model.index.id_map).tolist())

    if search.query.text is None and search.query.uid not in indexed_ids:
        search.query.text = normalize_documents([search.query.uid])

    for i, (id_, text) in enumerate(zip(ids, texts)):
        try:
            if text is None and id_ not in indexed_ids:
                texts[i] = normalize_documents([str(id_)])
        except HTTPException:
            # Some bogus PMID - set text as empty string
            logger.warning(f"Error encountered in normalize_documents: {id_}")
            texts[i] = ""

    # We then embed the corresponding text and update the index
    to_embed = [(id_, text) for id_, text in zip(ids, texts) if id_ not in indexed_ids]
    if to_embed:
        ids, texts = zip(*to_embed)  # type: ignore
        embeddings = encode(texts).cpu().numpy()  # type: ignore
        add_to_faiss_index(ids, embeddings, model.index)

    # Embed the query
    query_embedding = encode(search.query.text).cpu().numpy()  # type: ignore
    num_indexed = model.index.ntotal
    # Can't search for more items than exist in the index
    top_k = min(num_indexed, search.top_k)

    if search.docs_only:
        top_k = num_indexed

    # Perform the search
    top_k_scores, top_k_indicies = model.index.search(query_embedding, top_k)

    top_k_indicies = top_k_indicies.reshape(-1).tolist()
    top_k_scores = top_k_scores.reshape(-1).tolist()

    # Pick out results for the incoming ids in search.documents
    if search.docs_only:
        documents_positions = [top_k_indicies.index(id) for id in ids]
        top_k_indicies = ids
        top_k_scores = [top_k_scores[position] for position in documents_positions]

    if int(search.query.uid) in top_k_indicies:
        index = top_k_indicies.index(int(search.query.uid))
        del top_k_indicies[index], top_k_scores[index]

    response = [TopMatch(uid=uid, score=score) for uid, score in zip(top_k_indicies, top_k_scores)]
    return response
