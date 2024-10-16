#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/simplified_700x700"
A_DECK_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_normalized.nc"

# MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-7.340-13.300" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=6.950-10.350-11.200" "/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=8.500-9.610-12.300")
MODEL_DIR_NAMES=("/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/wavelengths-microns=3.900-6.185-6.950")
MEAN_TRANSLATION_DISTANCES_PX=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012" "013" "014" "015" "016" "017" "018" "019" "020" "021" "022" "023" "024" "025" "026" "027" "028" "029" "030" "031" "032" "033" "034" "035" "036" "037" "038" "039" "040" "041" "042" "043" "044" "045" "046" "047" "048" "049" "050" "051" "052" "053" "054" "055" "056" "057" "058" "059" "060" "061" "062" "063" "064" "065" "066" "067" "068" "069" "070" "071" "072" "073" "074" "075" "076" "077" "078" "079" "080" "081" "082" "083" "084" "085" "086" "087" "088" "089" "090" "091" "092" "093" "094" "095" "096" "097" "098")
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

CYCLONE_ID_STRING="2024AL14"
VALID_DATE_STRING="20241007"

for mean_translation_distance_px in "${MEAN_TRANSLATION_DISTANCES_PX[@]}"; do
    log_file_name="create_ensemble_mean-trans-dist-px=${mean_translation_distance_px}.out"

    python3 -u "${CODE_DIR_NAME}/create_multimodel_ensemble.py" &> ${log_file_name} \
    --input_prediction_file_names "${MODEL_DIR_NAMES[0]}/sensitivity_analysis_to_trans_dist/mean_translation_distance_px=${mean_translation_distance_px}/isotonic_regression/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc" "${MODEL_DIR_NAMES[1]}/sensitivity_analysis_to_trans_dist/mean_translation_distance_px=${mean_translation_distance_px}/isotonic_regression/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc" "${MODEL_DIR_NAMES[2]}/sensitivity_analysis_to_trans_dist/mean_translation_distance_px=${mean_translation_distance_px}/isotonic_regression/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc" "${MODEL_DIR_NAMES[3]}/sensitivity_analysis_to_trans_dist/mean_translation_distance_px=${mean_translation_distance_px}/isotonic_regression/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc" \
    --max_total_ensemble_size=10000 \
    --output_prediction_file_name="${ENSEMBLE_DIR_NAME}/sensitivity_analysis_to_trans_dist/mean_translation_distance_px=${mean_translation_distance_px}/isotonic_regression/predictions_${CYCLONE_ID_STRING}_${VALID_DATE_STRING}.nc" \
done
