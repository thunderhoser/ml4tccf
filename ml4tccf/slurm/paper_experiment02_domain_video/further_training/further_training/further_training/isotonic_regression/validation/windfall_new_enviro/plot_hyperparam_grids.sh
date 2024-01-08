#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_domain_video/further_training/further_training/further_training"

python3 -u "${CODE_DIR_NAME}/waf2024_plot_hyperparam_grids.py" \
--experiment_dir_name="${TOP_MODEL_DIR_NAME}" \
--use_isotonic_regression=1
