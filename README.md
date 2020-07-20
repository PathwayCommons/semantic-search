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

Once the server is running, you can make a POST request to with JSON body that is

1. Self-contained: Provide a `query` text and set of text `documents` to search against.  Sample JSON request:

    ```json
    {
        "query": {
            "uid":"9887103",
            "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development. The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms. These factors signal through a heteromeric complex of type I and II serine/threonine kinase receptors that phosphorylate members of the Smad family of transcription factors, thereby promoting their nuclear localization. Although components of TGF-beta/Activin signaling pathways are well defined in vertebrates, no such pathway has been clearly defined in invertebrates. In this study we describe the role of Baboon (Babo), a type I Activin receptor previously called Atr-I, in Drosophila development and characterize aspects of the Babo intracellular signal-transduction pathway. Genetic analysis of babo loss-of-function mutants and ectopic activation studies indicate that Babo signaling plays a role in regulating cell proliferation. In mammalian cells, activated Babo specifically stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive promoters but not BMP-responsive elements. Furthermore, we identify a new Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads 2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea, the Drosophila Smad4 homolog. Our results define a novel Drosophila Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show that this pathway functions to promote cellular growth with minimal effects on patterning."
        },
        "documents":[
            {
                "uid": "9887103",
                "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development. The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms. These factors signal through a heteromeric complex of type I and II serine/threonine kinase receptors that phosphorylate members of the Smad family of transcription factors, thereby promoting their nuclear localization. Although components of TGF-beta/Activin signaling pathways are well defined in vertebrates, no such pathway has been clearly defined in invertebrates. In this study we describe the role of Baboon (Babo), a type I Activin receptor previously called Atr-I, in Drosophila development and characterize aspects of the Babo intracellular signal-transduction pathway. Genetic analysis of babo loss-of-function mutants and ectopic activation studies indicate that Babo signaling plays a role in regulating cell proliferation. In mammalian cells, activated Babo specifically stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive promoters but not BMP-responsive elements. Furthermore, we identify a new Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads 2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea, the Drosophila Smad4 homolog. Our results define a novel Drosophila Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show that this pathway functions to promote cellular growth with minimal effects on patterning."
            },
            {
                "uid": "30049242",
                "text": "Transcriptional up-regulation of the TGF-β intracellular signaling transducer Mad of Drosophila larvae in response to parasitic nematode infection. The common fruit fly Drosophila melanogaster is an exceptional model for dissecting innate immunity. However, our knowledge on responses to parasitic nematode infections still lags behind. Recent studies have demonstrated that the well-conserved TGF-β signaling pathway participates in immune processes of the fly, including the anti-nematode response. To elucidate the molecular basis of TGF-β anti-nematode activity, we performed a transcript level analysis of different TGF-β signaling components following infection of D. melanogaster larvae with the nematode parasite Heterorhabditis gerrardi. We found no significant changes in the transcript level of most extracellular ligands in both bone morphogenic protein (BMP) and activin branches of the TGF-β signaling pathway between nematode-infected larvae and uninfected controls. However, extracellular ligand, Scw, and Type I receptor, Sax, in the BMP pathway as well as the Type I receptor, Babo, in the activin pathway were substantially up-regulated following H. gerrardi infection. Our results suggest that receptor up-regulation leads to transcriptional up-regulation of the intracellular component Mad in response to H. gerrardi following changes in gene expression of intracellular receptors of both TGF-β signaling branches. These findings identify the involvement of certain TGF-β signaling pathway components in the immune signal transduction of D. melanogaster larvae against parasitic nematodes ."
            },
            {
                "uid": "22936248",
                "text": "High-fidelity promoter profiling reveals widespread alternative promoter usage and transposon-driven developmental gene expression. Many eukaryotic genes possess multiple alternative promoters with distinct expression specificities. Therefore, comprehensively annotating promoters and deciphering their individual regulatory dynamics is critical for gene expression profiling applications and for our understanding of regulatory complexity. We introduce RAMPAGE, a novel promoter activity profiling approach that combines extremely specific 5'-complete cDNA sequencing with an integrated data analysis workflow, to address the limitations of current techniques. RAMPAGE features a streamlined protocol for fast and easy generation of highly multiplexed sequencing libraries, offers very high transcription start site specificity, generates accurate and reproducible promoter expression measurements, and yields extensive transcript connectivity information through paired-end cDNA sequencing. We used RAMPAGE in a genome-wide study of promoter activity throughout 36 stages of the life cycle of Drosophila melanogaster, and describe here a comprehensive data set that represents the first available developmental time-course of promoter usage. We found that >40% of developmentally expressed genes have at least two promoters and that alternative promoters generally implement distinct regulatory programs. Transposable elements, long proposed to play a central role in the evolution of their host genomes through their ability to regulate gene expression, contribute at least 1300 promoters shaping the developmental transcriptome of D. melanogaster. Hundreds of these promoters drive the expression of annotated genes, and transposons often impart their own expression specificity upon the genes they regulate. These observations provide support for the theory that transposons may drive regulatory innovation through the distribution of stereotyped cis-regulatory modules throughout their host genomes."
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

    NB: In this case, each `uid` in `documents` should be unique, but otherwise have no meaning.

2. References uids for PubMed articles. Sample JSON request:

    ```json
    {
        "query": "9887103",
        "documents":["9887103", "30049242", "22936248"],
        "top_k":3
    }
    ```

    NB:  You may use either objects (as in Case 1) or PMID strings for `query` and/or elements of `documents`. Howwever, the `documents` elements must all be of one type.

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
