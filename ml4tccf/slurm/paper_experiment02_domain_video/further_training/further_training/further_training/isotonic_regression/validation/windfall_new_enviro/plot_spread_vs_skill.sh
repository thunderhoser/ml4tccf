#!/bin/sh

model_dir_name=$1

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/plot_spread_vs_skill.py" \
--input_file_name="${model_dir_name}/isotonic_regression/validation/spread_vs_skill.nc" \
--output_dir_name="${model_dir_name}/isotonic_regression/validation/spread_vs_skill"
