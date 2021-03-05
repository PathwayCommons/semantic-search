import json

import hypothesis.strategies as st
import numpy as np
from fastapi.testclient import TestClient
from hypothesis import given, settings
from semantic_search import main
from semantic_search.main import app, app_startup, encode
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

client = TestClient(app)


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

    def test_query(self, dummy_requests: str) -> None:
        # dummy_requests fixutre returns list of all possible request types
        for request in dummy_requests:
            # Check that we can make a POST request with properly formatted payload
            response = client.post("/", request)
            assert response.status_code == 200

            # Check that the returned UIDs and scores are as expected
            expected_uids = [
                int(item["uid"]) if isinstance(item, dict) else int(item)
                for item in json.loads(request)["documents"]
            ]
            actual_uids = [item["uid"] for item in response.json()]
            actual_scores = [item["score"] for item in response.json()]
            del expected_uids[0]
            assert len(expected_uids) == len(actual_uids)
            assert set(actual_uids)  == set(expected_uids)
            assert all(0 <= score <= 1 for score in actual_scores)

    @settings(deadline=None)
    @given(bad_top_k=st.integers(max_value=0))
    def test_top_k_gt_zero(self, dummy_request_with_text: str, bad_top_k: int) -> None:
        # TODO: We can change this back to just using the dummy_requests fixture
        # once we solve the issue that causes requests with IDs to keep fetching the text
        # even if the ids have been indexed.
        dummy_requests = [dummy_request_with_text]
        for request in dummy_requests:
            request = json.loads(request)
            request["top_k"] = bad_top_k  # type: ignore
            request = json.dumps(request)
            response = client.post("/", request)
            assert response.status_code == 422
