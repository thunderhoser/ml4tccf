#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track/simplified_700x700"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DESCRIPTION_STRINGS=("3.900-7.340-13.300" "6.950-10.350-11.200" "8.500-9.610-12.300")

CYCLONE_ID_STRING="2024WP21"
VALID_DATE_STRINGS=("20241006" "20241007" "20241008" "20241009" "20241010" "20241011")
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration"

for valid_date_string in "${VALID_DATE_STRINGS[@]}"; do
    log_file_name="convert_for_zhixing_${CYCLONE_ID_STRING}_${valid_date_string}.out"

    python3 -u "${CODE_DIR_NAME}/convert_prediction_file_for_zhixing.py" &> ${log_file_name} \
    --input_prediction_file_name="${ENSEMBLE_DIR_NAME}/${CYCLONE_ID_STRING}_${valid_date_string}.nc" \
    --input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
    --output_prediction_dir_name="${ENSEMBLE_DIR_NAME}/for_zhixing"
done
