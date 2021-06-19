![build](https://github.com/PathwayCommons/semantic-search/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/PathwayCommons/semantic-search/branch/main/graph/badge.svg?token=K7444IQC9I)](https://codecov.io/gh/PathwayCommons/semantic-search)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/PathwayCommons/semantic-search?color=blue)

# Scientific Semantic Search

A simple semantic search engine for scientific papers. Check out our demo [here](https://share.streamlit.io/pathwaycommons/semantic-search/semantic_search/demo.py).

## Installation

This repository requires Python 3.7 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

If you _don't_ plan on modifying the source code, install from `git` using `pip`

```
pip install git+https://github.com/PathwayCommons/semantic-search.git
```

Otherwise, clone the repository locally and then install

```bash
git clone https://github.com/PathwayCommons/semantic-search.git
cd semantic-search
pip install --editable .
```

Finally, if you would like to take advantage of a CUDA-enabled GPU, you must also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

To start up the server:

```bash
uvicorn semantic_search.main:app
```

> You can pass the `--reload` flag if you are developing to force the server to reload on changes.

To provide arguments to the server, pass them as environment variables, e.g.:

```bash
CUDA_DEVICE=0 MAX_LENGTH=384 uvicorn semantic_search.main:app
```

Once the server is running, you can make a POST request to the `/search` endpoint with a JSON body. E.g.

```json
{
  "query": {
    "uid": "9887103",
    "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development."
  },
  "documents": [
    {
      "uid": "10320478",
      "text": "Drosophila dSmad2 and Atr-I transmit activin/TGFbeta signals. "
    },
    {
      "uid": "22563507",
      "text": "R-Smad competition controls activin receptor output in Drosophila. "
    },
    {
      "uid": "18820452",
      "text": "Distinct signaling of Drosophila Activin/TGF-beta family members. "
    },
    {
      "uid": "10357889"
    },
    {
      "uid": "31270814"
    }
  ],
  "top_k": 3
}
```

The return value is a JSON representation of the `top_k` most similar documents (defaults to 10):

```json
[
  {
    "uid": "10320478",
    "score": 0.6997108459472656
  },
  {
    "uid": "22563507",
    "score": 0.6877762675285339
  },
  {
    "uid": "18820452",
    "score": 0.6436074376106262
  }
]
```

If `"text"` is not provided, we assume `"uid"`s are valid PMIDs and fetch the title and abstract text before embedding, indexing and searching.

### Running via Docker

#### Setup

If you are intending on using a CUDA-enabled GPU, you must also install the NVIDIA Container Toolkit on the host following the instructions for your system [here](https://github.com/NVIDIA/nvidia-docker).

For Ubuntu 18.04:

```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```

Restart Docker

```
sudo systemctl stop docker
sudo systemctl start docker
```

Check your install

```
docker run --gpus all nvidia/cuda:10.2-cudnn7-devel nvidia-smi
```

#### Running a container

First, build the docker image:

```bash
docker build -t semantic-search .
```

Then, run it

```bash
docker run -it -p <PORT>:8000 semantic-search
```

For CUDA-enabled GPU

```
docker run --gpus all -dt --rm --name semantic_container -p 8000:8000 --env CUDA_DEVICE=0 --env MAX_LENGTH=384 semantic-search:latest
```

## Documentation

With the web server running, open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) in your browser for the API documentation.
