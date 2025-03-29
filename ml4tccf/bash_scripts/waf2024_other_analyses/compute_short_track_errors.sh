#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/waf2024"
PROCESSED_SHORT_TRACK_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/short_track_data/processed"
RAW_BEST_TRACK_DIR_NAME="/mnt/ssd-data1/temp_data/track_interpolated/best_track"

log_file_name="waf2024_compute_short_track_errors.out"

python3 -u "${CODE_DIR_NAME}/waf2024_compute_short_track_errors.py" &> ${log_file_name} \
--input_processed_short_track_dir_name="${PROCESSED_SHORT_TRACK_DIR_NAME}" \
--input_raw_best_track_dir_name="${RAW_BEST_TRACK_DIR_NAME}" \
--output_file_name="${PROCESSED_SHORT_TRACK_DIR_NAME}/short_track_errors_2024.nc"
