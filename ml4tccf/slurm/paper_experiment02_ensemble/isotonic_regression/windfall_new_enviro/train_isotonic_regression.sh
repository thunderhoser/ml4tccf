#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/train_scalar_iso_regression.py" \
--input_prediction_file_pattern="${model_dir_name}/training/predictions_*.nc" \
--output_model_dir_name="${model_dir_name}/isotonic_regression"
