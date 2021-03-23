from typing import Dict, List, Tuple


import numpy as np
from fastapi.testclient import TestClient

from semantic_search import main
from semantic_search.main import app, app_startup, encode
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

client = TestClient(app)

Request = Tuple[str, List[Dict[str, int]]]


class TestMain:
    app_startup()

    def test_encode(self, inputs) -> None:
        embeddings = encode(inputs)

        # These examples are hand chosen so that this is true.
        assert np.dot(embeddings[0], embeddings[1]) < np.dot(embeddings[2], embeddings[3])
        assert np.dot(embeddings[0], embeddings[2]) < np.dot(embeddings[0], embeddings[1])
        assert np.dot(embeddings[0], embeddings[3]) < np.dot(embeddings[0], embeddings[1])
        assert np.dot(embeddings[2], embeddings[0]) < np.dot(embeddings[2], embeddings[3])
        assert np.dot(embeddings[2], embeddings[1]) < np.dot(embeddings[2], embeddings[3])

    def test_setup_model_and_tokenizer(self) -> None:
        assert isinstance(main.model.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
        assert isinstance(main.model.model, PreTrainedModel)

    def test_query(self, dummy_requests: Request) -> None:
        # dummy_requests fixutre returns list of all possible request types
        for request, expected_response in dummy_requests:  # type: ignore
            # Check that we can make a POST request with properly formatted payload
            actual_response = client.post("/", request)  # type: ignore
            assert actual_response.status_code == 200

            # Check that the returned UIDs and scores are as expected
            expected_uids = [item["uid"] for item in expected_response]  # type: ignore
            actual_uids = [item["uid"] for item in actual_response.json()]
            actual_scores = [item["score"] for item in actual_response.json()]
            assert len(expected_uids) == len(actual_uids)
            assert set(actual_uids) == set(expected_uids)
            assert all(0 <= score <= 1 for score in actual_scores)
