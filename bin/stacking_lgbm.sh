
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
    python stacking_lgbm.py \
        store.model_name=lgbm_stacking \
        data.seed=777 \
        is_adversarial_validation=False \
        is_quantile=False \
        is_split_label=True
