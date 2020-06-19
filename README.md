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
`CUDA_DEVICE=0 BATCH_SIZE=64 MAX_LENGTH=384 uvicorn main:app`
```

Once the server is running, you can make a POST request with some query text and some documents to search against, and it will return the `top_k` most similar documents (if `top_k` is not provided, defaults to returning all documents), e.g.:

```bash
curl --header "Content-Type: application/json" --request POST --data '{"query":{"uid":"someid","text":"The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms."},"documents":[{"uid":"9887103","text":"The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development.\n"},{"uid":"30049242","text":"Transcriptional up-regulation of the TGF-β intracellular signaling transducer Mad of Drosophila larvae in response to parasitic nematode infection.\n"},{"uid":"22936248","text":"High-fidelity promoter profiling reveals widespread alternative promoter usage and transposon-driven developmental gene expression.\n"}],"top_k":3}' http://localhost:8000/
```

### Running via Docker

First, build the docker image:

```bash
docker build -t semantic-search .
```

Then, run it

```bash
docker run -it -p <PORT>:8000 semantic-search
```

## Documentation

With the web server running, open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) in your browser for the API documentation.
