import pytest

from semantic_search.ncbi import _medline_to_docs


def test_invalid_uid_test():
    with pytest.raises(TypeError):
        uid = ["93846392868"]
        _medline_to_docs(uid)
