# FROM docker.yard.oa.com:14917/yard/pytorch:cuda9.0-gpu-py36-from-source-0.4
# RUN pip install -i http://10.123.98.50/pypi/web/simple/ --trusted-host 10.123.98.50 torch==1.2.0 

FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y net-tools
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
RUN bash scripts/install_spacy.sh
RUN python scripts/install_nltk.py

# pydantic can cause llama index import fail, re-install
RUN pip uninstall -y pydantic 
RUN pip install pydantic
