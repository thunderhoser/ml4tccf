#!/bin/sh

model_dir_name=$1
cyclone_id_string=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/apply_scalar_iso_regression.py" \
--input_prediction_file_name="${model_dir_name}/testing/with_random_seed/predictions_${cyclone_id_string}.nc" \
--input_model_file_name="${model_dir_name}/isotonic_regression/isotonic_regression.dill" \
--output_prediction_file_name="${model_dir_name}/isotonic_regression/testing/with_random_seed/predictions_${cyclone_id_string}.nc"
