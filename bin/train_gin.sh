for is_train in true false; do
    if "${is_train}"; then
        warm_start=false
    else
        warm_start=true
    fi
    for i in 0 1 2 3 4
    do
    docker run --rm -it \
        -v $PWD/:/root/workdir/ \
        -v $HOME/.config/:/root/.config \
        -v $HOME/.netrc/:/root/.netrc \
        -v $HOME/.cache/:/root/.cache \
        -v $HOME/.git/:/root/.git \
        -e SLURM_LOCALID=0 \
        --shm-size=150gb \
        --runtime=nvidia \
        --ipc=host \
        --security-opt seccomp=unconfined \
        kaggle/pytorch:trends \
        python main_nn.py \
            base.opt_name=adam \
            data.n_fold=$i \
            data.is_train=$is_train \
            data.dataset_name=transformer_dataset \
            data.use_pseudo_label=False \
            train.warm_start=$warm_start \
            train.learning_rate=0.0005 \
            train.accumulation_steps=1 \
            train.refinement_step=8 \
            store.model_name=gin \
            store.save_feature=True \
            model.backbone=None \
            model.model_name=gin \
            model.in_channels=400 \
            model.dropout_rate=0.2 \
            train.batch_size=64 \
            train.epoch=25 \
            test.batch_size=8
    done
done