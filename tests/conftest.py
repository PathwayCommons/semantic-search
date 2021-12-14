import json
from typing import Dict, List, Tuple, Any

import pytest

Request = Tuple[str, List[Dict[str, Any]]]


@pytest.fixture(scope="module")
def inputs() -> List[str]:
    # Some random examples where sentence 1-2 and 3-4 are most similar to each other.
    return [
        "The inhibition of AICAR suppresses the phosphorylation of TBC1D1.",
        "TBC1D1 phosphorylation is increased by AICAR, but only responds minimally to contraction.",
        "Ras and Mek are in proximity, and they phosphorylate ASPP2.",
        "Ras and Mek are in proximity, and ASPP2 phosphorylates them.",
    ]


@pytest.fixture(scope="module")
def dummy_request_with_test() -> Request:
    request = {
        "query": {
            "uid": "9887103",
            "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls...",
        },
        "documents": [
            {
                "uid": "9887103",
                "text": "The Drosophila activin receptor baboon signals through dSmad2 and...",
            },
            {
                "uid": "30049242",
                "text": "Transcriptional up-regulation of the TGF-β intracellular signaling...",
            },
            {
                "uid": "22936248",
                "text": "High-fidelity promoter profiling reveals widespread alternative...",
            },
        ],
        "top_k": 3,
    }
    # We don't actually test scores, so use a dummy value of -1
    response = [{"uid": "30049242", "score": -1}, {"uid": "22936248", "score": -1}]
    return json.dumps(request), response


@pytest.fixture(scope="module")
def followup_request_with_test() -> Request:
    request = {
        "query": {
            "uid": "9813169",
            "text": "TGF-beta signaling from the cell surface to the nucleus is mediated by the SMAD...",
        },
        "documents": [
            {
                "uid": "10320478",
                "text": "Much is known about the three subfamilies of the TGFbeta superfamily in vertebrates...",
            },
            {
                "uid": "10357889",
                "text": "The transforming growth factor-beta (TGF-beta) superfamily encompasses a large...",
            },
            {
                "uid": "15473904",
                "text": "Members of TGFbeta superfamily are found to play important roles in many cellular...",
            },
        ],
        "docs_only": True,
    }
    # We don't actually test scores, so use a dummy value of -1
    response = [
        {"uid": "10320478", "score": -1},
        {"uid": "10357889", "score": -1},
        {"uid": "15473904", "score": -1},
    ]
    return json.dumps(request), response
