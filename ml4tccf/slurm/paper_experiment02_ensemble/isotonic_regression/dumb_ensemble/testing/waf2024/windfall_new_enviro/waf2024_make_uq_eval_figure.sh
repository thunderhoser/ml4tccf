#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_ensemble/tropical_only"

python3 -u "${CODE_DIR_NAME}/waf2024_make_uq_eval_figure.py" \
--input_evaluation_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/validation" \
--output_figure_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/validation/waf2024/uq_eval_figure"
