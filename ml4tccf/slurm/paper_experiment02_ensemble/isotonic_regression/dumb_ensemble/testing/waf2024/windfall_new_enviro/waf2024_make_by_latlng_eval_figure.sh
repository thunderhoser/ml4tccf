#!/bin/sh

evaluation_dir_name=$1
output_dir_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/waf2024_make_2category_eval_figure.py" \
--input_evaluation_dir_name="${evaluation_dir_name}" \
--output_figure_dir_name="${output_dir_name}"
