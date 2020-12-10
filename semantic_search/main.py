from operator import itemgetter
from typing import Callable, Dict, List, Optional, Tuple, cast

import torch
import typer
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings, validator
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from semantic_search.ncbi import uids_to_docs

PRETRAINED_MODEL = "johngiorgi/declutr-sci-base"

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
SUCCESS = "\U00002705"
WARNING = "\U000026A0"
FAST = "\U0001F3C3"


app = FastAPI()


class Settings(BaseSettings):
    """Store global settings for the web-service. Pass these as environment variables at server
    startup. E.g.

    `CUDA_DEVICE=0 MAX_LENGTH=384 uvicorn semantic_search.main:app`
    """

    pretrained_model_name_or_path: str = PRETRAINED_MODEL
    batch_size: int = 64
    max_length: Optional[int] = None
    mean_pool: bool = True
    cuda_device: int = -1


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None
    similarity: Callable[..., torch.Tensor] = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    uid: str
    text: str = None


class Query(BaseModel):
    query: Document
    documents: List[Document] = []
    top_k: Optional[int] = None

    @validator("query", "documents")
    def normalize_document(cls, v, field):
        if field.name == "query":
            v = [v]

        normalized_docs = []
        for doc in v:
            if doc.uid is None and doc.text is None:
                raise ValueError(f'Got None for both the "uid" and "text" in {field}.')
            if doc.text is None:
                normalized_docs.append(Document(**uids_to_docs([doc.uid])[0]))
            else:
                normalized_docs.append(doc)
        return normalized_docs[0] if field.name == "query" else normalized_docs


settings = Settings()
model = Model()


def _get_device(cuda_device):
    """Return a `torch.cuda` device if `torch.cuda.is_available()` and `cuda_device>=0`.
    Otherwise returns a `torch.cpu` device.
    """
    if cuda_device != -1 and torch.cuda.is_available():
        device = torch.device("cuda")
        typer.secho(
            f"{FAST} Using CUDA device {torch.cuda.get_device_name()} with index {torch.cuda.current_device()}.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        device = torch.device("cpu")
        typer.secho(
            f"{WARNING} Using CPU. Note that this will be many times slower than a GPU.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    return device


def _setup_model_and_tokenizer(
    pretrained_model_name_or_path: str, cuda_device: int = -1
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    device = _get_device(cuda_device)
    # Load the Transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    typer.secho(
        f'{SUCCESS} Tokenizer "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )
    # Load the Transformers model
    model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device)
    model.eval()
    typer.secho(
        f'{SUCCESS} Model "{pretrained_model_name_or_path}" from Transformers loaded successfully.',
        fg=typer.colors.GREEN,
        bold=True,
    )

    return tokenizer, model


@torch.no_grad()
def _encode(
    text: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    mean_pool: bool = True,
) -> torch.Tensor:

    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=settings.max_length, return_tensors="pt"
    )
    for name, tensor in inputs.items():
        inputs[name] = tensor.to(model.device)
    attention_mask = inputs["attention_mask"]
    output = model(**inputs).last_hidden_state

    if mean_pool:
        embedding = torch.sum(output * attention_mask.unsqueeze(-1), dim=1) / torch.clamp(
            torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9
        )
    else:
        embedding = output[:, 0, :]

    return embedding


def encode(text: List[str]) -> torch.Tensor:
    # Sort the inputs by length, maintaining the original indices so we can un-sort
    # before returning the embeddings. This speeds up embedding by minimizing the
    # amount of computation performed on pads. Because this sorting happens before
    # tokenization, it is only a proxy of the true lengths of the inputs to the model.
    # In the future, it would be better to sort by length *after* tokenization which
    # would lead to an even larger speedup.
    # https://stackoverflow.com/questions/8372399/zip-with-list-output-instead-of-tuple
    sorted_indices, text = cast(
        Tuple[Tuple[int], Tuple[str]], zip(*sorted(enumerate(text), key=itemgetter(1)))
    )  # tell mypy explicitly the types of items in the unpacked tuple
    unsorted_indices, _ = zip(*sorted(enumerate(sorted_indices), key=itemgetter(1)))

    embeddings: torch.Tensor = []
    for i in range(0, len(text), settings.batch_size):
        embedding = _encode(
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

    model.tokenizer, model.model = _setup_model_and_tokenizer(
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
