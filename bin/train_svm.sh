
docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --shm-size=64gb \
    --runtime=nvidia \
    --ipc=host \
    --security-opt seccomp=unconfined \
    kaggle/pytorch:trends \
    python main_svm.py store.model_name=svm_rbf

docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --shm-size=64gb \
    --runtime=nvidia \
    --ipc=host \
    --security-opt seccomp=unconfined \
    kaggle/pytorch:trends \
    python main_svm.py store.model_name=svm_linear kernel=linear


