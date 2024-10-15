#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/simplified_700x700"
A_DECK_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_normalized.nc"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DESCRIPTION_STRINGS=("3.900-6.185-6.950" "3.900-7.340-13.300" "6.950-10.350-11.200" "8.500-9.610-12.300")

CYCLONE_ID_STRING="2024EP99"
VALID_DATE_STRINGS=("20241007" "20241008" "20241009")
i=-1

for model_dir_name in "${MODEL_DIR_NAMES[@]}"; do
    i=$((i + 1))

    for valid_date_string in "${VALID_DATE_STRINGS[@]}"; do
        log_file_name="apply_isotonic_regression_${CYCLONE_ID_STRING}_${MODEL_DESCRIPTION_STRINGS[$i]}_${valid_date_string}.out"
    
        python3 -u "${CODE_DIR_NAME}/apply_scalar_iso_regression.py" &> ${log_file_name} \
        --input_model_file_name="${model_dir_name}/isotonic_regression/isotonic_regression.dill" \
        --input_prediction_file_name="${model_dir_name}/real_time_predictions/predictions_${CYCLONE_ID_STRING}_${valid_date_string}.nc" \
        --output_prediction_file_name="${model_dir_name}/real_time_predictions/isotonic_regression/${CYCLONE_ID_STRING}_${valid_date_string}.nc"
    done
done
