FROM python:3

ADD . /

RUN pip3 install "fastapi[all]"
RUN pip3 install -r requirements.txt

ENTRYPOINT ["uvicorn","main:app","--host","0.0.0.0"]
