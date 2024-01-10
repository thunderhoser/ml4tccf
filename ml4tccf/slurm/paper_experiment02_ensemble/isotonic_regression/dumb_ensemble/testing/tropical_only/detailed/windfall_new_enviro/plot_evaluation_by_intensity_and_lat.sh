#!/bin/sh

max_wind_cutoffs_string=$1
eval_file_names_string=$2
output_dir_name=$3

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"

python3 -u "${CODE_DIR_NAME}/plot_evaluation_by_intensity.py" \
--max_wind_cutoffs_kt ${max_wind_cutoffs_string} \
--tc_center_latitude_cutoffs_deg_n 15 25 35 45 \
--input_max_wind_based_eval_file_names ${eval_file_names_string} \
--label_font_size=40 \
--confidence_level=0.95 \
--output_dir_name="${output_dir_name}"
