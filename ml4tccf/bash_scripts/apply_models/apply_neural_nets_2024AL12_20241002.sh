#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/simplified_700x700"
A_DECK_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_normalized.nc"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")

CYCLONE_ID_STRING="2024AL12"
VALID_DATE_STRING="20241002"
LOG_FILE_NAME="apply_neural_nets_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.out"

for model_dir_name in "${MODEL_DIR_NAMES[@]}"; do
    if [[ "$model_dir_name" == "${MODEL_DIR_NAMES[0]}" ]]; then
        python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" &> ${LOG_FILE_NAME} \
        --input_model_file_name="${model_dir_name}/model.h5" \
        --input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
        --input_a_deck_file="${A_DECK_FILE_NAME}" \
        --cyclone_id_string="${CYCLONE_ID_STRING}" \
        --valid_date_string="${VALID_DATE_STRING}" \
        --data_aug_num_translations=8 \
        --random_seed=6695 \
        --synoptic_times_only=0 \
        --disable_gpus=1 \
        --output_file_name="${model_dir_name}/predictions/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc"
    else
        python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" &>> ${LOG_FILE_NAME} \
        --input_model_file_name="${model_dir_name}/model.h5" \
        --input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
        --input_a_deck_file="${A_DECK_FILE_NAME}" \
        --cyclone_id_string="${CYCLONE_ID_STRING}" \
        --valid_date_string="${VALID_DATE_STRING}" \
        --data_aug_num_translations=8 \
        --random_seed=6695 \
        --synoptic_times_only=0 \
        --disable_gpus=1 \
        --output_file_name="${model_dir_name}/predictions/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc"
    fi
done
