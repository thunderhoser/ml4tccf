#!/bin/bash

CODE_DIR_NAME="/home/ralager/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_ensemble/tropical_only"
EXAMPLE_DIR_NAME="/home/ralager/condo/swatwork/ralager/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_for_paper/simplified_all_wavelengths"

cyclone_id_string="2017AL01"
wavelength_group_string_microns="6.950-10.350-11.200"

model_dir_name="${TOP_MODEL_DIR_NAME}/wavelengths-microns=${wavelength_group_string_microns}"
echo $model_dir_name

time ~/anaconda3/bin/python3.7 -u "${CODE_DIR_NAME}/apply_neural_net_2gpus.py" \
--input_model_file_name="${model_dir_name}/model.h5" \
--input_satellite_dir_name="${EXAMPLE_DIR_NAME}" \
--cyclone_id_string="${cyclone_id_string}" \
--num_bnn_iterations=1 \
--max_ensemble_size=100 \
--data_aug_num_translations=1 \
--output_dir_name="${model_dir_name}/timing_tests/1translation"
