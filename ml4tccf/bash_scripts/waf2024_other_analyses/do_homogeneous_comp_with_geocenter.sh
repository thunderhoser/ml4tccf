#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/waf2024"
RAW_ARCHER_FILE_PATTERN="/mnt/ssd-data1/temp_data/ARCHER_2024/2024*.txt"
RAW_BEST_TRACK_DIR_NAME="/mnt/ssd-data1/temp_data/track_interpolated/best_track"
GEOCENTER_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration/for_zhixing"
OUTPUT_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration/homogeneous_comparison_with_geocenter2.nc"

log_file_name="waf2024_do_homogeneous_comp_with_geocenter.out"

python3 -u "${CODE_DIR_NAME}/waf2024_do_homogeneous_comp_with_geocenter.py" &> ${log_file_name} \
--input_raw_archer_file_pattern="${RAW_ARCHER_FILE_PATTERN}" \
--input_raw_best_track_dir_name="${RAW_BEST_TRACK_DIR_NAME}" \
--input_geocenter_dir_name="${GEOCENTER_DIR_NAME}" \
--output_file_name="${OUTPUT_FILE_NAME}"
