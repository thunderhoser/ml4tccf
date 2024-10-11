#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/simplified_700x700"

CYCLONE_ID_STRING="2024EP99"
LOG_FILE_NAME="simplify_satellite_files_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/simplify_satellite_files.py" &> ${LOG_FILE_NAME} \
--input_dir_name="${INPUT_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--wavelengths_to_keep_microns 3.9 6.185 6.95 7.34 8.5 9.61 10.35 11.2 12.3 13.3 \
--num_grid_rows_to_keep=700 \
--num_grid_columns_to_keep=700 \
--output_dir_name="${OUTPUT_DIR_NAME}"
