from operator import itemgetter
from typing import List, Optional, Tuple, Union, cast

import faiss
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
from semantic_search.schemas import Model, Query, Response

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


@app.post("/", response_model=List[Response])
async def query(query: Query):
    ids = [int(doc.uid) for doc in query.documents]
    texts = [document.text for document in query.documents]

    # # Ensure that the query is not in the index when we search.
    # query_id = np.asarray(int(query.query.uid)).reshape(
    #     1,
    # )
    # model.index.remove_ids(query_id)

    # Only add items to the index if they do not already exist.
    # See: https://github.com/facebookresearch/faiss/issues/859
    # To do this, we first determine which of the incoming ids do not exist in the index
    indexed_ids = set(faiss.vector_to_array(model.index.id_map).tolist())
    to_embed = [(id_, text) for id_, text in zip(ids, texts) if id_ not in indexed_ids]
    # We then embed the corresponding text and update the index
    if to_embed:
        ids, texts = zip(*to_embed)  # type: ignore
        embeddings = encode(texts).cpu().numpy()
        add_to_faiss_index(ids, embeddings, model.index)

    # Can't search for more items than exist in the index
    top_k = min(model.index.ntotal, query.top_k + 1)

    # Embed the query and perform the search
    query_embedding = encode(query.query.text).cpu().numpy()
    top_k_scores, top_k_indicies = model.index.search(query_embedding, top_k)

    top_k_indicies = top_k_indicies.reshape(-1).tolist()
    top_k_scores = top_k_scores.reshape(-1).tolist()

    if int(query.query.uid) in top_k_indicies:
        index = top_k_indicies.index(int(query.query.uid))
        del top_k_indicies[index], top_k_scores[index]
    else:
        del top_k_indicies[-1], top_k_scores[-1]

    response = [Response(uid=uid, score=score) for uid, score in zip(top_k_indicies, top_k_scores)]
    return response
