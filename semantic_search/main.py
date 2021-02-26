import os
from operator import itemgetter
from typing import Dict, List, Optional, Tuple, cast

import torch
from fastapi import FastAPI
from pydantic import BaseSettings

from semantic_search import __version__
from semantic_search.common.util import (
    add_to_faiss_index,
    encode_with_transformer,
    setup_faiss_index,
    setup_model_and_tokenizer,
)
from semantic_search.schemas import Model, Query

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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


def encode(text: List[str]) -> torch.Tensor:
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


@app.post("/")
async def query(query: Query) -> List[Dict[str, float]]:
    text = [query.query.text] + [document.text for document in query.documents]

    embeddings = encode(text)

    # query_id = query.query.uid
    query_embedding = embeddings[0].unsqueeze(0).cpu().numpy()
    document_ids = [int(doc.uid) for doc in query.documents]
    document_embeddings = embeddings[1:].cpu().numpy()

    add_to_faiss_index(document_ids, document_embeddings, model.index)

    # Ensure that the query is not in the index when we search.
    # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#removing-elements-from-an-index
    # model.index.remove_ids(query_id)

    if query.top_k is not None:
        top_k = max(min(query.top_k, len(query.documents)), 0)
    top_k_scores, top_k_indicies = model.index.search(query_embedding, top_k)

    # model.index.add_with_ids(query_embedding, query_id)

    top_k_indicies = top_k_indicies.reshape(-1).tolist()
    top_k_scores = top_k_scores.reshape(-1).tolist()

    return [{"uid": uid, "score": score} for uid, score in zip(top_k_indicies, top_k_scores)]
