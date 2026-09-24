#!/bin/bash
# Diagnosis classification on MIMIC with kNN probes on frozen EchoTab measurement tokens.

export ECHOTAB_ROOT=${ECHOTAB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_DIR=/path/to/mimic/echo-tab
ECHOTAB_RUN_DIR=/path/to/training_logs/EchoTab-Pretrain/<experiment_name>

feature_source="measurement_tokens" # or measurement_tokens_all

for diagnosis in "aortic_stenosis" "aortic_regurg" "mitral_regurg" "la_size" "ra_size" "lv_size" "rv_size" "lvef_class" "rv_function"; do
    python $ECHOTAB_ROOT/run.py --config-name=base_knn \
        data_dir=$DATA_DIR \
        echotab_run_dir=$ECHOTAB_RUN_DIR \
        downstream_task=diagnosis_ordinal_classification_mimic \
        downstream_task.target=$diagnosis \
        downstream_task.feature_source=$feature_source \
        downstream_task.paths.type=_balanced \
        validate=true \
        test=true
done
