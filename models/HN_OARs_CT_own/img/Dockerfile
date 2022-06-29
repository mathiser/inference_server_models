FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /
RUN pip install -e /nnUNet

ENV nnUNet_raw_data_base=/Task/nnUNet_raw_data_base/
ENV nnUNet_preprocessed=/Task/nnUNet_preprocessed/
ENV nnUNet_raw_data=/Task/nnUNet_raw_data_base/nnUNet_raw_data/
ENV RESULTS_FOLDER=/Task/nnUNet_trained_models/

ENV INPUT=/input
ENV OUTPUT=/output
ENV RESULTS_FOLDER=/model

RUN mkdir $INPUT $OUTPUT $RESULTS_FOLDER


RUN ["chmod", "+x", "/main.sh"]

CMD ["bash", "-c", "/main.sh"]
