#!/bin/bash
# EchoTab pretraining on MIMIC with the hyperparameters used in the paper
# (all of them are the defaults in configs/; listed here for visibility).

export ECHOTAB_ROOT=${ECHOTAB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_DIR=/path/to/mimic/echo-tab

python $ECHOTAB_ROOT/run.py --config-name=base_pretrain_echotab \
    data_dir=$DATA_DIR \
    max_epochs=30 \
    dataset.batch_size=4 \
    dataset.accum_iter=16 \
    model.echo.preserve_local_tokens=true \
    model.echo.use_bg_removal=true \
    dataset.echo.background_threshold=-0.6 \
    model.tab_decoder.num_layers=2 \
    model.tab_decoder.num_heads=4 \
    model.tab_decoder.layerscale_init=1 \
    model.prediction_head.head_type=partial_shared \
    model.prediction_head.num_head_layers=2 \
    model.prediction_head.hypernet_size=M \
    model.prediction_head.lora_rank=8 \
    model.prediction_head.lora_alpha=16 \
    model.prediction_head.cond_proj_dim=128 \
    model.prediction_head.hyper_hidden=256 \
    model.prediction_head.use_single_hypernet=true \
    train.l2_loss.use_var_per_task=true \
    train.params.lr_echo_branch=5e-5 \
    train.params.lr=1e-4 \
    train.params.weight_decay=0.0 \
    train.gradient_clip_val=50
