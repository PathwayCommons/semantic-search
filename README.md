# Scientific Semantic Search

A simple semantic search engine for scientific papers.

## Installation

This repository requires Python 3.7 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally:

```bash
git clone https://github.com/BaderLab/semantic-search.git
cd semantic-search
```

Then, install

```bash
pip install -e .
```

Finally, if you would like to take advantage of a CUDA-enabled GPU, you must also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

To start up the server:

```bash
uvicorn main:app
```

> You can pass the `--reload` flag if you are developing to force the server to reload on changes.

To provide arguments to the server, pass them as environment variables, e.g.:

```bash
CUDA_DEVICE=0 MAX_LENGTH=384 uvicorn main:app
```

Once the server is running, you can make a POST request with:

1. JSON body that is self-contained. Provide the text in `query` and text in `documents` to search against. Sample JSON request:

    ```json
    {
        "query": {
            "uid":"9887103",
            "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development. The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms..."
        },
        "documents":[
            {
                "uid": "9887103",
                "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development. The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms..."
            },
            {
                "uid": "30049242",
                "text": "Transcriptional up-regulation of the TGF-β intracellular signaling transducer Mad of Drosophila larvae in response to parasitic nematode infection. The common fruit fly Drosophila melanogaster is an exceptional model for dissecting innate immunity..."
            },
            {
                "uid": "22936248",
                "text": "High-fidelity promoter profiling reveals widespread alternative promoter usage and transposon-driven developmental gene expression. Many eukaryotic genes possess multiple alternative promoters with distinct expression specificities..."
            }
        ],
        "top_k":3
    }
    ```

    The return value is a JSON representation of the `top_k` most similar documents (default: return all):

    ```json
    [
        {
            "uid": "9887103",
            "score": 1.0
        },
        {
            "uid": "30049242",
            "score": 0.6427373886108398
        },
        {
            "uid": "22936248",
            "score": 0.49102723598480225
        }
    ]
    ```

    - NB: In this case, each `uid` in `documents` should be unique, but otherwise have no meaning.

2. JSON body that references PubMed article uids. Sample JSON request:

    ```json
    {
        "query": "9887103",
        "documents":["9887103", "30049242", "22936248"],
        "top_k":3
    }
    ```

    - Notes:
      - For each Document element, the text consists of the `ArticleTitle` appended to `Abstract` for that PubMed article. See [pubmed DTD](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/180101/index.html)
      - JSON body may consist of either objects (as in Case 1) or PMID strings for `query` and elements of `documents`. However, the elements of `documents` must either be all be a single type.

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
