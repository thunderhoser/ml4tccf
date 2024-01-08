#!/bin/sh

eval_file_names_string=$1
output_dir_name=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/plot_evaluation_by_xy_coords.py" \
--x_coord_cutoffs_metres -3000000 -1500000 0 1500000 3000000 4500000 \
--y_coord_cutoffs_metres 1500000 3000000 4500000 \
--input_evaluation_file_names ${eval_file_names_string} \
--confidence_level=0.95 \
--label_font_size=30 \
--output_dir_name="${output_dir_name}"
