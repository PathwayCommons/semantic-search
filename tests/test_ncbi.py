import pytest

from fastapi.exceptions import HTTPException

from semantic_search.ncbi import _medline_to_docs


def test_invalid_uid_test():
    with pytest.raises(HTTPException):
        uid = ["93846392868"]
        records = [{"id:": [uid]}]
        _medline_to_docs(records)
