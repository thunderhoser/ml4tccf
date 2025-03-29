#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"

CYCLONE_ID_STRING="2024EP10"

YEAR=${CYCLONE_ID_STRING:0:4}
REST=${CYCLONE_ID_STRING:4}
REST_LOWER=${REST,,}
FAKE_CYCLONE_ID_STRING="${REST_LOWER}${YEAR}"

ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble/real_time_predictions_short_track/isotonic_regression_gaussian_dist/uncertainty_calibration"

log_file_name="compare_with_zhixing_${CYCLONE_ID_STRING}.out"

python3 -u "${CODE_DIR_NAME}/plot_ryan_zhixing_comparison.py" &> ${log_file_name} \
--input_ryan_dir_name="${ENSEMBLE_DIR_NAME}/for_zhixing/${FAKE_CYCLONE_ID_STRING}" \
--input_zhixing_dir_name="${ENSEMBLE_DIR_NAME}/for_zhixing/${FAKE_CYCLONE_ID_STRING}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--input_raw_best_track_file_name="/mnt/shceph/data/ATCF/atcf_rt/ATCF/dat/NHC/b${FAKE_CYCLONE_ID_STRING}.dat" \
--output_file_name="${ENSEMBLE_DIR_NAME}/for_zhixing/${CYCLONE_ID_STRING}.jpg"
