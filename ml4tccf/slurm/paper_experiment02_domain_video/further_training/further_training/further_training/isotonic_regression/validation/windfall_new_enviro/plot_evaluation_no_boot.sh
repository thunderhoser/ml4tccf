#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/plot_scalar_evaluation.py" \
--input_evaluation_file_name="${model_dir_name}/isotonic_regression/validation/evaluation_no_bootstrap.nc" \
--confidence_level=0.95 \
--output_dir_name="${model_dir_name}/isotonic_regression/validation/evaluation_no_bootstrap"
