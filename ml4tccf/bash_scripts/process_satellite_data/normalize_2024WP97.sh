#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper"
NORMALIZATION_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/ir_satellite_normalization_params.zarr"

CYCLONE_ID_STRING="2024WP97"
LOG_FILE_NAME="normalize_satellite_data_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/normalize_satellite_data.py" &> ${LOG_FILE_NAME} \
--input_satellite_dir_name="${INPUT_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--input_normalization_file_name="${NORMALIZATION_FILE_NAME}" \
--output_satellite_dir_name="${OUTPUT_DIR_NAME}"
