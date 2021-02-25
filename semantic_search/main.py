from operator import itemgetter
from typing import Dict, List, Optional, Tuple, cast

import torch
from fastapi import FastAPI
from pydantic import BaseSettings

from semantic_search.common.util import encode_with_transformer, setup_model_and_tokenizer
from semantic_search.schemas import Model, Query
from semantic_search import __version__

app = FastAPI(
    title="Scientific Semantic Search",
    description="A simple semantic search engine powered by HuggingFace's Transformers library.",
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
    model.similarity = torch.nn.CosineSimilarity(-1)


@app.post("/")
async def query(query: Query) -> List[Dict[str, float]]:
    ids = [query.query.uid] + [document.uid for document in query.documents]
    text = [query.query.text] + [document.text for document in query.documents]

    embeddings = encode(text)
    similarity_scores = model.similarity(embeddings[0], embeddings[1:])

    # If top_k not specified, return all documents.
    top_k = similarity_scores.size(0)
    if query.top_k is not None:
        top_k = max(min(query.top_k, len(query.documents)), 0)
    top_k_scores, top_k_indicies = torch.topk(similarity_scores, top_k)
    top_k_scores = top_k_scores.tolist()
    # Offset the indices by 1 to account for the query
    top_k_indicies = [idx.item() + 1 for idx in top_k_indicies]

    return [{"uid": ids[idx], "score": top_k_scores[num]} for num, idx in enumerate(top_k_indicies)]
