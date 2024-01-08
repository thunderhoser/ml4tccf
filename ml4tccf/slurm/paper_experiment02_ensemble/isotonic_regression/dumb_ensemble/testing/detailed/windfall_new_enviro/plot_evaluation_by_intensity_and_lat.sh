#!/bin/sh

eval_file_names_string=$1
output_dir_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/plot_evaluation_by_intensity.py" \
--max_wind_cutoffs_kt 50 80 \
--tc_center_latitude_cutoffs_deg_n 15 25 35 45 \
--input_max_wind_based_eval_file_names ${eval_file_names_string} \
--label_font_size=40 \
--confidence_level=0.95 \
--output_dir_name="${output_dir_name}"
