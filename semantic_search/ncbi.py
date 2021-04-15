import io
import os
from pathlib import Path
from typing import Any, Dict, List, Generator
import logging
import json
import time

import requests
from Bio import Medline
from dotenv import load_dotenv
from pydantic import BaseSettings
from fastapi import HTTPException

log = logging.getLogger(__name__)


def _compact(input: List) -> List:
    """Returns a list with None, False, and empty String removed"""
    return [x for x in input if x is not None and x is not False and x != ""]


# -- Setup and initialization --
MAX_EFETCH_RETMAX = 10000
dot_env_filepath = Path(__file__).absolute().parent.parent / ".env"
load_dotenv(dot_env_filepath)


class Settings(BaseSettings):
    app_name: str = os.getenv("APP_NAME", "")
    app_version: str = os.getenv("APP_VERSION", "")
    app_url: str = os.getenv("APP_URL", "")
    admin_email: str = os.getenv("ADMIN_EMAIL", "")
    ncbi_eutils_api_key: str = os.getenv("NCBI_EUTILS_API_KEY", "")
    eutils_base_url: str = os.getenv("EUTILS_BASE_URL", "")
    eutils_efetch_url: str = eutils_base_url + os.getenv("EUTILS_EFETCH_BASENAME", "")
    eutils_esummary_url: str = eutils_base_url + os.getenv("EUTILS_ESUMMARY_BASENAME", "")
    http_request_timeout: int = int(os.getenv("HTTP_REQUEST_TIMEOUT", -1))


settings = Settings()


# -- NCBI EUTILS --
def _safe_request(url: str, method: str = "GET", headers={}, **opts):
    user_agent = f"{settings.app_name}/{settings.app_version} ({settings.app_url};mailto:{settings.admin_email})"
    request_headers = {"user-agent": user_agent}
    request_headers.update(headers)
    try:
        r = requests.request(
            method, url, headers=request_headers, timeout=settings.http_request_timeout, **opts
        )
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout error {e}")
        raise
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error {e}; status code: {r.status_code}")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in request {e}")
        raise
    else:
        return r


def _parse_medline(text: str) -> List[dict]:
    """Convert the rettype=medline to dict.
    See https://www.nlm.nih.gov/bsd/mms/medlineelements.html
    """
    f = io.StringIO(text)
    medline_records = Medline.parse(f)
    return medline_records


def _get_eutil_records(eutil: str, id: List[str], **opts) -> List[Dict[Any, Any]]:
    """Call one of the NCBI EUTILITIES and returns data as Python objects."""
    eutils_params = {
        "db": "pubmed",
        "id": ",".join(id),
        "retstart": 0,
        "retmode": "xml",
        "api_key": settings.ncbi_eutils_api_key,
    }
    eutils_params.update(opts)
    if eutil == "esummary":
        url = settings.eutils_esummary_url
    elif eutil == "efetch":
        url = settings.eutils_efetch_url
    else:
        raise ValueError(f"Unsupported eutil '{eutil}''")
    eutilResponse = _safe_request(url, "POST", files=eutils_params)
    return _parse_medline(eutilResponse.text)


def _medline_to_docs(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a list Documents given a list of Medline records
    See https://www.nlm.nih.gov/bsd/mms/medlineelements.html
    """
    docs = []
    for record in records:
        if "PMID" not in record:
            raise HTTPException(status_code=422, detail=f"No PMID for {json.dumps(record)}") 
        pmid = record["PMID"]
        abstract = record["AB"] if "AB" in record else ""
        title = record["TI"] if "TI" in record else ""
        text = " ".join(_compact([title, abstract]))
        docs.append({"uid": pmid, "text": text})
    return docs


# -- Public methods --
def uids_to_docs(uids: List[str]) -> Generator[List[Dict[str, str]], None, None]:
    """Return uid, and text (i.e. title + abstract) given a PubMed uid"""
    num_uids = len(uids)
    num_queries = num_uids // MAX_EFETCH_RETMAX + 1
    for i in range(num_queries):
        lower = i * MAX_EFETCH_RETMAX
        upper = min([lower + MAX_EFETCH_RETMAX, num_uids])
        id = uids[lower:upper]
        try:
            start_time = time.time()
            eutil_response = _get_eutil_records("efetch", id, rettype="medline", retmode="text")
            duration = time.time() - start_time
            logging.info(
                f"Retrieved docs {lower} through {upper - 1} of {num_uids - 1} in {duration}s"
            )
        except Exception as e:
            logging.warn(f"Error encountered in uids_to_docs: {e}")
            logging.warn(f"Bypassing docs {lower} through {upper - 1} of {num_uids - 1}")
            continue
        else:
            yield _medline_to_docs(eutil_response)
