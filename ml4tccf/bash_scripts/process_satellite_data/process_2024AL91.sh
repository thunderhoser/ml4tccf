#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
INPUT_DIR_NAME="/mnt/shceph/users2/tccdb/nrt_geocenter/g16/0.0.7/2024"
OUTPUT_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed"

CYCLONE_ID_STRING="2024AL91"
LOG_FILE_NAME="process_satellite_data_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/process_satellite_data.py" &> ${LOG_FILE_NAME} \
--input_dir_name="${INPUT_DIR_NAME}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--max_bad_pixels_low_res=6250 \
--max_bad_pixels_high_res=25000 \
--temporary_dir_name="${OUTPUT_DIR_NAME}/temporary_${CYCLONE_ID_STRING}" \
--min_visible_pixel_fraction=0.9 \
--altitude_angle_exe_name="/home/lagerquist/solarpos/solarpos" \
--output_dir_name="${OUTPUT_DIR_NAME}"
