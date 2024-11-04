#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/subset"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/subset/recentered"
SHORT_TRACK_DIR_NAME="/mnt/shnas10/users/galka/track_interpolated/short_track"

CYCLONE_ID_STRING="2024AL09"
LOG_FILE_NAME="recenter_satellite_data_${CYCLONE_ID_STRING}_test.out"

python3 -u "${CODE_DIR_NAME}/recenter_satellite_data_on_short_track.py" &> ${LOG_FILE_NAME} \
--input_satellite_dir_name="${INPUT_DIR_NAME}" \
--input_short_track_dir_name="${SHORT_TRACK_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--output_satellite_dir_name="${OUTPUT_DIR_NAME}"
