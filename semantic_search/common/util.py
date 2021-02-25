from enum import Enum
from typing import Tuple, List, Optional

import torch
import typer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class Emoji(Enum):
    # Emoji's used in typer.secho calls
    # See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py
    SUCCESS = "\U00002705"
    WARNING = "\U000026A0"
    FAST = "\U0001F3C3"


def get_device(cuda_device: int = -1) -> torch.device:
    """Return a `torch.cuda` device if `torch.cuda.is_available()` and `cuda_device>=0`.
    Otherwise returns a `torch.cpu` device.
    """
    if cuda_device != -1 and torch.cuda.is_available():
        device = torch.device("cuda")
        typer.secho(
            f"{Emoji.FAST.value} Using CUDA device {torch.cuda.get_device_name()} with index"
            f" {torch.cuda.current_device()}.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        device = torch.device("cpu")
        typer.secho(
            f"{Emoji.WARNING.value} Using CPU. Note that this will be many times slower than a GPU.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    return device


def setup_model_and_tokenizer(
    pretrained_model_name_or_path: str, cuda_device: int = -1
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Given a HuggingFace Transformer `pretrained_model_name_or_path`, return the corresponding
    model and tokenizer. Optionally, places the model on `cuda_device`, if available.
    """
    device = get_device(cuda_device)
    # Load the Transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    typer.secho(
        (
            f'{Emoji.SUCCESS.value} Tokenizer "{pretrained_model_name_or_path}" from Transformers'
            " loaded successfully."
        ),
        fg=typer.colors.GREEN,
        bold=True,
    )
    # Load the Transformers model
    model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device)
    model.eval()
    typer.secho(
        (
            f'{Emoji.SUCCESS.value} Model "{pretrained_model_name_or_path}" from Transformers'
            " loaded successfully."
        ),
        fg=typer.colors.GREEN,
        bold=True,
    )

    return tokenizer, model


@torch.no_grad()
def encode_with_transformer(
    text: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_length: Optional[int] = None,
    mean_pool: bool = True,
) -> torch.Tensor:

    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
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
