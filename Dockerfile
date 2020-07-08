FROM nvidia/cuda:10.2-base
CMD nvidia-smi

FROM python:3

ADD . /

RUN pip install -e .

ENTRYPOINT ["uvicorn","main:app","--host","0.0.0.0"]
