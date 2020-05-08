# Scientific Semantic Search

A simple semantic search engine for scientific papers.

## Installation

This repository requires Python 3.7 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally

```bash
git clone https://github.com/BaderLab/semantic-search.git
cd semantic-search
```

Then, install the requirements

```bash
pip install "fastapi[all]"
pip install -r requirements.txt
```

## Usage

You will need to prepare an index of text to search against, this is simply a text file with one text instance per line ([abstracts.txt](abstracts.txt) is provided for demo purposes). Then, start up the server with the path to this index:


```bash
INDEX_FILEPATH="abstracts.txt"  uvicorn main:app
```

Then you can make a POST request with some query text, and it will return the `top_k` most similar articles from the index:

```bash
curl \
 --data "text=Pathway switching in target cells is a previously unreported mechanism for regulating TGFbeta signaling." \
 http://127.0.0.1:8000/query
```

## Documentation

With the web server running, open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) in your browser for the API documentation.
