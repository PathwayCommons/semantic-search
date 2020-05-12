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

To start up the server:


```bash
uvicorn main:app
```

> You can pass the `--reload` flag if you are developing to force the server to reload on changes.

You can then make a POST request with some query text and some documents to search against, and it will return the `top_k` most similar documents.


## Documentation

With the web server running, open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) in your browser for the API documentation.
