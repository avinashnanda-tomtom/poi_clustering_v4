# FROM ubuntu:18.04
FROM rapidsai/rapidsai:cuda11.2-runtime-ubuntu18.04-py3.8

# Install base utilities
RUN apt-get update
RUN apt-get install -y htop vim
RUN apt-get install build-essential -y


COPY ./enviroment.yml .
COPY ./requirements.txt .


RUN source activate rapids \
    && pip install -r requirements.txt


RUN source activate rapids \
    && conda env update --file enviroment.yml

RUN TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=$PYTHONPATH:/workspace
ENV PYTHONPATH=$PYTHONPATH:/workspace/clustering
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/rapids/lib
