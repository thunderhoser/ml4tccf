#!/bin/sh

model_dir_name=$1
cyclone_id_string=$2

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_for_paper/800x800_grids"

python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
--input_model_file_name="${model_dir_name}/model.h5" \
--input_satellite_dir_name="${EXAMPLE_DIR_NAME}" \
--cyclone_id_string="${cyclone_id_string}" \
--num_bnn_iterations=1 \
--max_ensemble_size=100 \
--data_aug_num_translations=8 \
--output_dir_name="${model_dir_name}/validation"
