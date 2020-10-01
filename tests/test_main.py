import numpy as np
from semantic_search import main
from semantic_search.main import app_startup, encode
from transformers import RobertaModel, RobertaTokenizer


class TestMain:
    app_startup()

    def test_setup_model_and_tokenizer(self) -> None:
        assert isinstance(main.model.tokenizer, RobertaTokenizer)
        assert isinstance(main.model.model, RobertaModel)

    def test_encode(self, inputs):
        embeddings = encode(inputs)
        # The first two sentences should be considered most similar.
        assert np.dot(embeddings[0], embeddings[1]) > np.dot(embeddings[0], embeddings[2])
        assert np.dot(embeddings[0], embeddings[1]) > np.dot(embeddings[1], embeddings[2])