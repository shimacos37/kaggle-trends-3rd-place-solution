FROM gcr.io/tpu-pytorch/xla:nightly
RUN apt-get update \
     && apt-get install --yes libsm6 \
     libxext6 \
     libxrender-dev 


RUN  pip install opencv-python \
     albumentations==0.4.3 \
     addict \
     setuptools \
     tqdm \
     scikit-image \
     future \
     facenet-pytorch \
     pretrainedmodels \
     hydra-core \
     hydra_colorlog


RUN mkdir worlkdir
WORKDIR /root/workdir


