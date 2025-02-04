#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/shnas10/data/SHORT_TRACK"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/short_track_data/processed"

CYCLONE_ID_STRING="2020EP09"
LOG_FILE_NAME="process_short_track_data_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/process_short_track_data.py" &> ${LOG_FILE_NAME} \
--input_raw_dir_name="${INPUT_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--num_minutes_back=360 \
--num_minutes_ahead=350 \
--output_processed_dir_name="${OUTPUT_DIR_NAME}"
