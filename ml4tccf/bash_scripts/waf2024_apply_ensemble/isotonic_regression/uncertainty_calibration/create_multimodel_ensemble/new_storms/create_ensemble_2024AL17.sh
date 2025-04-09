#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DESCRIPTION_STRINGS=("3.900-7.340-13.300" "6.950-10.350-11.200" "8.500-9.610-12.300")

CYCLONE_ID_STRING="2024AL17"
VALID_DATE_STRINGS=("20241102" "20241103" "20241104")
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration"

for valid_date_string in "${VALID_DATE_STRINGS[@]}"; do
    log_file_name="create_ensemble_${CYCLONE_ID_STRING}_${valid_date_string}.out"

    python3 -u "${CODE_DIR_NAME}/create_multimodel_ensemble.py" &> ${log_file_name} \
    --input_prediction_file_names "${MODEL_DIR_NAMES[0]}/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration/${CYCLONE_ID_STRING}_${valid_date_string}.nc" "${MODEL_DIR_NAMES[1]}/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration/${CYCLONE_ID_STRING}_${valid_date_string}.nc" "${MODEL_DIR_NAMES[2]}/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration/${CYCLONE_ID_STRING}_${valid_date_string}.nc" \
    --max_total_ensemble_size=10000 \
    --output_prediction_file_name="${ENSEMBLE_DIR_NAME}/${CYCLONE_ID_STRING}_${valid_date_string}.nc"
done
