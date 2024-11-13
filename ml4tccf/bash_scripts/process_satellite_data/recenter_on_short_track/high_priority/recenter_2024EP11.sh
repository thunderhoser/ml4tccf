#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track"
SHORT_TRACK_DIR_NAME="/mnt/shnas10/users/galka/track_interpolated/short_track"

CYCLONE_ID_STRING="2024EP11"
LOG_FILE_NAME="recenter_satellite_data_on_short_track_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/recenter_satellite_data_on_short_track.py" &> ${LOG_FILE_NAME} \
--input_satellite_dir_name="${INPUT_DIR_NAME}" \
--input_short_track_dir_name="${SHORT_TRACK_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--output_satellite_dir_name="${OUTPUT_DIR_NAME}"
