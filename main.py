from typing import List, Tuple

import torch
import typer
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
SUCCESS = "\U00002705"

app = FastAPI()


class Settings(BaseSettings):
    """Store global settings for the web-service. Pass these as enviornment variables at server
    startup. E.g.

    `MEAN_POOL=True uvicorn main:app`
    """

    pretrained_model_name_or_path: str = "allenai/scibert_scivocab_uncased"
    batch_size: int = 32
    mean_pool: bool = False
    cuda_device: int = -1


class Model(BaseModel):
    tokenizer: PreTrainedModel = None
    model: PreTrainedTokenizer = None

    class Config:
        arbitrary_types_allowed = True


class Index(BaseModel):
    text: List[str] = None
    ids: List[str] = None
    embeddings: torch.Tensor = None

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    uid: str
    text: str


class Query(BaseModel):
    query: Document
    documents: List[Document] = []
    top_k: int = None


settings = Settings()
model = Model()
index = Index()


def _get_device(cuda_device):
    """Return a `torch.cuda` device if `torch.cuda.is_available()` and `cuda_device>=0`.
    Otherwise returns a `torch.cpu` device.
    """
    if cuda_device != -1 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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


def _index(text: List[str]) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(text), settings.batch_size):
        embedding = _encode(
            text[i : i + settings.batch_size],
            tokenizer=model.tokenizer,
            model=model.model,
            mean_pool=settings.mean_pool,
        )
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings)
    return embeddings


@torch.no_grad()
def _encode(
    text: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    mean_pool: bool = False,
):
    inputs = tokenizer.batch_encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    sequence_output, _ = model(**inputs)

    if mean_pool:
        embedding = torch.sum(
            sequence_output * inputs["attention_masks"].unsqueeze(-1), dim=1
        ) / torch.clamp(torch.sum(inputs["attention_masks"], dim=1, keepdims=True), min=1e-9)
    else:
        embedding = sequence_output[:, 0, :]

    return embedding


@app.on_event("startup")
def app_startup():

    model.tokenizer, model.model = _setup_model_and_tokenizer(
        settings.pretrained_model_name_or_path, cuda_device=settings.cuda_device
    )


@app.post("/")
async def query(query: Query):
    embedded_query = _encode(
        [query.query.text], model.tokenizer, model.model, mean_pool=settings.mean_pool
    )

    text = [document.text for document in query.documents]
    ids = [document.uid for document in query.documents]
    embeddings = _index(text)
    index.text = text
    index.ids = ids
    index.embeddings = embeddings

    cos = torch.nn.CosineSimilarity(-1)
    similarity_scores = cos(embedded_query, index.embeddings)
    # If top_k not specified, return all documents.
    top_k = query.top_k or similarity_scores.size(0)
    top_k_scores, top_k_indicies = torch.topk(similarity_scores, top_k)

    top_k_scores = top_k_scores.tolist()
    top_k_indicies = top_k_indicies.tolist()

    return [
        {"uid": index.ids[idx], "score": top_k_scores[num]}
        for num, idx in enumerate(top_k_indicies)
    ]
