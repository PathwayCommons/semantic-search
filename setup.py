import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="semantic-search",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("A simple semantic search engine powered by HuggingFace's Transformers library."),
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.6.0",
    install_requires=[
        "fastapi>=0.61.1",
        "uvicorn>=0.12.1",
        "torch>=1.6.0",
        "transformers>=3.3.1",
        "typer>=0.3.2",
        "python-dotenv>=0.14.0",
        "xmltodict>=0.12.0",
    ],
    extras_require={"dev": ["black", "coverage", "codecov", "flake8", "pytest", "pytest-cov"]},
)
