#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track/simplified_700x700"
A_DECK_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_normalized.nc"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DESCRIPTION_STRINGS=("3.900-6.185-6.950" "3.900-7.340-13.300" "6.950-10.350-11.200" "8.500-9.610-12.300")

CYCLONE_ID_STRING="2024EP11"
VALID_DATE_STRINGS=("20241001" "20241002" "20241003" "20241004")
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

for valid_date_string in "${VALID_DATE_STRINGS[@]}"; do
    log_file_name="convert_for_zhixing_${CYCLONE_ID_STRING}_${valid_date_string}.out"

    python3 -u "${CODE_DIR_NAME}/convert_prediction_file_for_zhixing.py" &> ${log_file_name} \
    --input_prediction_file_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/${CYCLONE_ID_STRING}_${valid_date_string}.nc" \
    --input_satellite_dir_name="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track/simplified_700x700" \
    --output_prediction_dir_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/for_zhixing"
done
