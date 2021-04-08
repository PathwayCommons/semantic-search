import streamlit as st
import requests
import json

URL = "https://semanticsearch.baderlab.org/"

st.sidebar.markdown(" # Scientific Semantic Search")
st.sidebar.markdown(
    "A demo for [BaderLab's Scientific Semantic Search tool](https://github.com/PathwayCommons/semantic-search)."
)
st.sidebar.markdown(
    "Entery a query PubMed Identifier (PMID) on the right and the most similar articles will be returned."
)
st.sidebar.markdown(
    "Similarity is determined using a [deep neural network](https://huggingface.co/johngiorgi/declutr-sci-base) which operates on the articles title and abstract."
)

st.sidebar.markdown("## Additional settings")

st.sidebar.markdown("### Top K")

# Add a slider to the sidebar:
top_k = st.sidebar.slider("Select the number of documents to include in the results", 1, 100, (10))

st.sidebar.markdown("### PMIDs to Include")
document_ids = st.sidebar.text_input(
    "Here you can enter PMIDs seperated by a space. These articles will be included in the search."
)
document_ids = [doc_id.strip() for doc_id in document_ids.strip().split()]

query_pmid = st.text_input("Enter a query PMID here")
query_pmid = query_pmid.strip()

query = {"query": query_pmid, "documents": document_ids, "top_k": top_k}
json_body = json.dumps(query)

if query_pmid:
    response = requests.post(URL, json_body)

    st.markdown("## Results")
    st.write("---")

    for item in response.json():
        link = f"https://pubmed.ncbi.nlm.nih.gov/{item['uid']}"
        st.write(f"__PMID__: [{item['uid']}]({link})")
        st.write(f"__Score__:{item['score']:.4}")
        st.write("---")
