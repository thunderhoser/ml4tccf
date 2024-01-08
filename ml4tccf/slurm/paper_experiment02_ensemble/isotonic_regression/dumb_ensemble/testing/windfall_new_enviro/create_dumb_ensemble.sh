#!/bin/sh

input_file_names_string=$1
output_file_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/create_multimodel_ensemble.py" \
--input_prediction_file_names ${input_file_names_string} \
--max_total_ensemble_size=1000 \
--output_prediction_file_name="${output_file_name}"
