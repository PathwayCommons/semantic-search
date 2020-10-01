from typing import List

import pytest


@pytest.fixture(scope="module")
def inputs() -> List[str]:
    # Some random examples where sentence 1 and 2 are most similar.
    return [
        "The inhibition of AICAR suppresses the phosphorylation of TBC1D1.",
        "TBC1D1 phosphorylation is increased by AICAR, but only responds minimally to contraction.",
        "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.",
    ]
