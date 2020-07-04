FROM rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.7

RUN apt-get update \
     && apt-get install --yes  --no-install-recommends \
     cuda-toolkit-10-2 \
     gcc \
     ssh-client \
     && apt-get clean

RUN ln -s /usr/local/cuda/bin/nvcc /opt/conda/envs/rapids/bin/ \
     && ln -s /usr/local/cuda/bin/nvcc.profile /opt/conda/envs/rapids/bin/ \
     && ln -s  /usr/local/cuda/nvvm /opt/conda/envs/rapids/
ENV CPATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$CPATH

RUN  source activate rapids \
     && pip install --ignore-installed opencv-python \
     albumentations==0.3.3 \
     tqdm \
     joblib \
     torch \
     torchvision \
     efficientnet-pytorch \
     pretrainedmodels \
     segmentation-models-pytorch \
     pytorch-lightning==0.7.5 \
     hydra-core \
     hydra_colorlog \
     google-cloud-storage \
     seaborn \
     plotly \
     nilearn \
     h5py \
     wandb \
     jupyterlab \
     jupyterlab_code_formatter \
     black \
     mypy \
     lightgbm \
     flake8 \
     timm \
     xgboost \
     rising \
     threadpoolctl \
     && git clone https://github.com/NVIDIA/apex \
     && cd apex \
     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--pyprof" . \
     && cd .. \ 
     && rm -r apex

RUN source activate rapids \
     && conda install -y -c dglteam dgl-cuda10.2

RUN source activate rapids\
     && conda install -y nodejs \ 
     && jupyter labextension install @lckr/jupyterlab_variableinspector \
     && jupyter labextension install @jupyterlab/toc \
     && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
     && jupyter serverextension enable --py jupyterlab_code_formatter \
     && jupyter labextension install jupyterlab-plotly

RUN mkdir worlkdir
WORKDIR /root/workdir