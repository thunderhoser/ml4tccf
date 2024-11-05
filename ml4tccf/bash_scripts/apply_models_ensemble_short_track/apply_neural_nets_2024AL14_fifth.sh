#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track/simplified_700x700"
A_DECK_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_normalized.nc"

MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DESCRIPTION_STRINGS=("3.900-6.185-6.950" "3.900-7.340-13.300" "6.950-10.350-11.200" "8.500-9.610-12.300")

CYCLONE_ID_STRING="2024AL14"
VALID_DATE_STRINGS=("20241005" "20241006" "20241007" "20241008" "20241009" "20241010")

for ((i = 4; i < 5; i++)); do
    for ((j = 0; j < ${#MODEL_DIR_NAMES[@]}; j++)); do
        log_file_name="apply_neural_nets_${CYCLONE_ID_STRING}_${MODEL_DESCRIPTION_STRINGS[$j]}_${VALID_DATE_STRINGS[$i]}.out"
    
        python3 -u "${CODE_DIR_NAME}/apply_neural_net_real_time.py" &> ${log_file_name} \
        --input_model_file_name="${MODEL_DIR_NAMES[$j]}/model.h5" \
        --input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
        --input_a_deck_file="${A_DECK_FILE_NAME}" \
        --cyclone_id_string="${CYCLONE_ID_STRING}" \
        --valid_date_string="${VALID_DATE_STRINGS[$i]}" \
        --disable_gpus=1 \
        --data_aug_num_translations=100 \
        --data_aug_num_translations_per_step=20 \
        --data_aug_mean_translation_low_res_px=24 \
        --data_aug_stdev_translation_low_res_px=12 \
        --random_seed=6695 \
        --output_file_name="${MODEL_DIR_NAMES[$j]}/ensemble_predictions_short_track/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRINGS[$i]}.nc"
    done
done
