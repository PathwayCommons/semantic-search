import streamlit as st
import requests
import json

URL = "https://semanticsearch.baderlab.org/"

st.sidebar.header("Scientific Semantic Search")
st.sidebar.write(
    "A demo for [BaderLab's Scientific Semantic Search tool](https://github.com/PathwayCommons/semantic-search)."
)
st.sidebar.write(
    "Entery a query PubMed Identifier (PMID) on the right and the most similar articles will be returned."
)
st.sidebar.write(
    (
        "Similarity is determined using a [deep neural network](https://huggingface.co/johngiorgi/declutr-sci-base)"
        " which operates on the articles title and abstract."
    )
)

st.sidebar.markdown("## Additional settings")

st.sidebar.subheader("Top K")

# Add a slider to the sidebar:
top_k = st.sidebar.slider(
    "Select the number of documents to include in the results", 1, 100, value=(10)
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
query_pmid = st.text_input("Enter a query PMID here", help="E.g., 33024307")
query_pmid = query_pmid.strip()

query = {"query": query_pmid, "documents": document_ids, "top_k": top_k}
json_body = json.dumps(query)

if query_pmid:
    response = requests.post(URL, json_body)

    """
    # Results
    ---
    """

    for item in response.json():
        link = f"https://pubmed.ncbi.nlm.nih.gov/{item['uid']}"
        st.write(f"__PMID__: [{item['uid']}]({link})")
        st.write(f"__Score__:{item['score']:.4}")
        st.write("---")
