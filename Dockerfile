FROM tiangolo/python-machine-learning:cuda9.1-python3.7

ADD . /

RUN pip install -e .

CMD ["uvicorn","main:app","--host","0.0.0.0"]
