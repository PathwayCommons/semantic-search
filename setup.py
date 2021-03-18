import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semantic-search",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("A simple semantic search engine for scientific papers."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PathwayCommons/semantic-search",
    packages=setuptools.find_packages(),
    keywords=["natural language processing", "pytorch", "transformers", "semantic search"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.7.0",
    install_requires=[
        "biopython>=1.78",
        "fastapi>=0.63.0",
        "faiss-cpu>=1.7.0",
        "uvicorn>=0.13.4",
        "torch>=1.7.1",
        "transformers>=4.3.3",
        "typer>=0.3.2",
        "python-dotenv>=0.15.0",
        "xmltodict>=0.12.0",
    ],
    extras_require={
        "dev": [
            "black",
            "coverage",
            "codecov",
            "flake8",
            "pytest",
            "pytest-cov",
            "hypothesis",
            "mypy",
        ]
    },
)
