import os
import re
from dotenv import load_dotenv
from pydantic import BaseSettings, BaseModel, validator
from typing import List, Dict, Optional, Any, Union
import requests
import xmltodict

# -- Setup and initialization --
MAX_EFETCH_RETMAX = 10000
dir_path = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(dir_path, '.env'))
ncbi_pubdate_pattern = re.compile(r"^(?P<year>\d{4})(\s(?P<month>[a-zA-Z\-]+))?(\s(?P<day>\d{1,2}))?$")
ncbi_month_pattern = re.compile(r"^[a-zA-Z]{3}$")
year_pattern = re.compile(r"(?P<year>\d{4})")
number_pattern = re.compile(r"^[0-9]+$")


class Settings(BaseSettings):
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")
    app_url = os.getenv("APP_URL")
    admin_email = os.getenv("ADMIN_EMAIL")
    ncbi_eutils_api_key = os.getenv("NCBI_EUTILS_API_KEY")
    eutils_base_url = os.getenv("EUTILS_BASE_URL")
    eutils_efetch_url = eutils_base_url + os.getenv("EUTILS_EFETCH_BASENAME")
    eutils_esummary_url = eutils_base_url + os.getenv("EUTILS_ESUMMARY_BASENAME")
    http_request_timeout = int(os.getenv("HTTP_REQUEST_TIMEOUT"))


settings = Settings()


# -- Utilities --
def _get(input: Dict, path: List, default=None) -> Any:
    """Returns the value for the path, which specifies the key (index) of the next Dict (List)
    """
    internal_dict_value = input
    for k in path:
        if number_pattern.match(k):
            internal_dict_value = internal_dict_value[int(k)]
        else:
            internal_dict_value = internal_dict_value.get(k, None)
        if internal_dict_value is None:
            return default
    return internal_dict_value


def _compact(input: List) -> List:
    """Returns a list with None, False, and empty String removed
    """
    return [x for x in input if x is not None and x is not False and x != ""]


# -- Models --
def get_element_text(cls, v):
    return v if isinstance(v, str) else v["#text"]


class PubDate(BaseModel):
    Year: str = ""
    Month: str = ""
    Day: str = ""
    Season: Optional[str]
    MedlineDate: Optional[str]

    @validator('MedlineDate')
    def populate_year(cls, v, values):
        year_match = year_pattern.match(v)
        if year_match:
            Year = year_match.groupdict()["year"]
            values["Year"] = Year
        return v


class JournalIssue(BaseModel):
    Volume: str = ""
    Issue: str = ""
    PubDate: PubDate


class Journal(BaseModel):
    Title: str = ""
    ISOAbbreviation: str = ""
    JournalIssue: JournalIssue


def label_and_text_from_element(element: Union[str, dict]):
    if isinstance(element, str):
        return element
    elif isinstance(element, dict):
        text = ""
        Label = _get(element, ["@Label"], "")
        if Label:
            text += Label + ": "
        text += _get(element, ["#text"], "")
        return text


class Abstract(BaseModel):
    AbstractText: Union[str, dict, List[Union[dict, str]]]

    # Can be str, dict, or list of dict/str
    @validator("AbstractText")
    def combine_labels_and_texts(cls, v):
        update = ""
        if isinstance(v, str) or isinstance(v, dict):
            update += label_and_text_from_element(v)
        else:
            for element in v:
                if update:
                    update += " "
                update += label_and_text_from_element(element)
        return update


class Article(BaseModel):
    Journal: Journal
    ArticleTitle: Union[str, dict]
    Abstract: Optional[Abstract]

    # validators
    _textify_article_title = validator('ArticleTitle', allow_reuse=True)(get_element_text)

    @validator("Abstract")
    def merge_abstract_text(cls, v):
        return v.AbstractText


class MedlineCitation(BaseModel):
    PMID: dict
    Article: Article

    # validators
    _textify_pmid = validator('PMID', allow_reuse=True)(get_element_text)


class PubmedArticle(BaseModel):
    MedlineCitation: MedlineCitation


class PubmedArticleSet(BaseModel):
    PubmedArticle: Union[PubmedArticle, List[PubmedArticle]]


class PubmedEfetchResponse(BaseModel):
    PubmedArticleSet: PubmedArticleSet


# -- NCBI EUTILS --
def _safe_request(url: str, method: str = 'GET', headers={}, **opts):
    user_agent = f"{settings.app_name}/{settings.app_version} ({settings.app_url};mailto:{settings.admin_email})"
    request_headers = {
        "user-agent": user_agent
    }
    request_headers.update(headers)
    try:
        r = requests.request(method, url, headers=request_headers, timeout=settings.http_request_timeout, **opts)
        r. raise_for_status()
    except requests.exceptions.Timeout as e:
        print(f"Timeout error {e}")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error {e}; status code: {r.status_code}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error in request {e}")
        raise
    else:
        return r


def _get_eutil_records(eutil: str, id: List[str], **opts) -> dict:
    """Call one of the NCBI EUTILITIES and returns data as Python objects.
    """
    eutils_params = {
        "db": "pubmed",
        "id": ",".join(id),
        "retstart": 0,
        "retmode": "xml",
        "api_key": settings.ncbi_eutils_api_key
    }
    eutils_params.update(opts)
    if eutil == "esummary":
        url = settings.eutils_esummary_url
    elif eutil == "efetch":
        url = settings.eutils_efetch_url
    else:
        raise ValueError(f"Unsupported eutil '{eutil}''")
    eutilResponse = _safe_request(url, 'POST', files=eutils_params)
    doc = xmltodict.parse(eutilResponse.text)
    return doc


def _articles_to_docs(articles: List[PubmedArticle]) -> List[Dict[str, str]]:
    """Return a list Documents given a PubmedArticle
    """
    docs = []
    for pubmed_article in articles:
        abstract = pubmed_article.MedlineCitation.Article.Abstract
        title = pubmed_article.MedlineCitation.Article.ArticleTitle
        text = " ".join(_compact([title, abstract]))
        pmid = pubmed_article.MedlineCitation.PMID
        docs.append({
            "uid": pmid,
            "text": text
        })
    return docs


# -- Public methods --
def uids_to_docs(uids: List[str]) -> List[Dict[str, str]]:
    """Return uid, and text (i.e. title + abstract) given a PubMed uid
    """
    docs = []
    num_uids = len(uids)
    num_queries = num_uids // MAX_EFETCH_RETMAX + 1
    for i in range(num_queries):
        lower = i * MAX_EFETCH_RETMAX
        upper = min([lower + MAX_EFETCH_RETMAX, num_uids])
        id = uids[lower:upper]
        print(f"Retrieving uids between {lower} and {upper}")
        try:
            eutil_response = _get_eutil_records('efetch', id)
            ERROR = _get(eutil_response, ["eFetchResult", "ERROR"])
            if ERROR:
                raise RuntimeError(ERROR)
            pubmedEfetchReponse = PubmedEfetchResponse(**eutil_response)
            pubmedArticle = pubmedEfetchReponse.PubmedArticleSet.PubmedArticle
        except Exception as e:
            print(f"Error encountered in uids_to_docs {e}")
            raise e
        else:
            articles = pubmedArticle if isinstance(pubmedArticle, list) else [pubmedArticle]
            output = _articles_to_docs(articles)
            docs = docs + output
    return docs
