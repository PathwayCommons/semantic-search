import json
from typing import List

import requests  # type: ignore
import streamlit as st
import validators

SEMANTIC_SEARCH_URL = "https://semanticsearch.baderlab.org/"
PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"


@st.cache(show_spinner=False)
def get_titles(pmids: List[str]):
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"
    json_body = {"pmids": pmids, "concepts": "none"}
    r = requests.post(url, json=json_body)
    articles = r.text.strip().split("\n\n")
    titles = {}
    for article in articles:
        pmid, title = article.split("\n")[0].split("|t|")
        titles[pmid] = title
    return titles


def display_paper(title: str, pmid: str, score: float) -> None:
    link = f"{PUBMED_BASE_URL}{pmid}"
    col1, col2, col3 = st.beta_columns(3)
    col1.write(f"__{title}__")
    col2.write(f"PMID: [{pmid}]({link})")
    col3.write(f"Score: {result['score']:.4}")
    st.write("---")


st.sidebar.write(
    """
    # Scientific Semantic Search

    A demo for [BaderLab's Scientific Semantic Search tool](https://github.com/PathwayCommons/semantic-search).

    Enter a query PubMed Identifier (PMID) on the right and the most similar articles will be returned.

    Similarity is determined using a [deep neural network](https://huggingface.co/johngiorgi/declutr-sci-base)
    which operates on the articles title and abstract.

    ## Additional settings
    """
)


st.sidebar.subheader("Top K")
top_k = st.sidebar.slider(
    "Select the number of documents to include in the results", 1, 50, value=(10)
)


st.sidebar.subheader("PMIDs to Include")
document_ids = []
document_ids.extend(
    st.sidebar.text_input(
        "Here you can enter PMIDs seperated by a space. These articles will be included in the search."
    )
    .strip()
    .split()
)

uploaded_file = st.sidebar.file_uploader(
    "Alternatively (or additionally), you can enter a file which contains one PMID per line."
)
if uploaded_file:
    document_ids.extend(uploaded_file.read().decode("utf-8").strip().split("\n"))
document_ids = [doc_id.strip() for doc_id in document_ids if doc_id]

"""
# Search
"""
query_pmid = st.text_input(
    "Enter the PMID or PubMed URL of a paper to query with", help="E.g., 33024307"
)
query_pmid = query_pmid.strip()
# Here, we determine if the user copy/pasted a URL and try to parse the PMID out if so.
# This could likely be handled better with a regex.
if validators.url(query_pmid):
    query_pmid = query_pmid.split(PUBMED_BASE_URL)[-1].replace("/", "").strip()
query = {"query": query_pmid, "documents": document_ids, "top_k": top_k}
json_body = json.dumps(query)

if query_pmid:
    response = requests.post(SEMANTIC_SEARCH_URL, json_body)

    pmids = [query_pmid] + [item["uid"] for item in response.json()]
    titles = get_titles(pmids=pmids)

    st.write("# Results")
    st.write(
        (
            f"Displaying the top {top_k} articles most similar to:"
            f" [__{titles[query_pmid]}__]({PUBMED_BASE_URL}{query_pmid})"
        )
    )
    st.write("---")

    for result in response.json():
        # The cast to str is because the current, deployed version of semantic search
        # is casting all PMIDs to ints, but this is a bug and will be fixed soon.
        pmid, score = str(result["uid"]), result["score"]
        title = titles.get(pmid, "Couldn't find title.")
        display_paper(title, pmid, score)
