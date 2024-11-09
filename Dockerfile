FROM python:3.10.6-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY API API/

COPY setup.py setup.py
RUN pip install -e .

RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y

CMD uvicorn API.Backend.fast_api.api:app --port=$PORT --host=0.0.0.0
