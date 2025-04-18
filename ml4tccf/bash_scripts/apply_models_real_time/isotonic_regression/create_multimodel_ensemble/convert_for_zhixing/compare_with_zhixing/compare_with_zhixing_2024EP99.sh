#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"

CYCLONE_ID_STRING="2024EP99"
FAKE_CYCLONE_ID_STRING="ep992024"
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

log_file_name="compare_with_zhixing_${CYCLONE_ID_STRING}_${valid_date_string}.out"

python3 -u "${CODE_DIR_NAME}/plot_ryan_zhixing_comparison.py" &> ${log_file_name} \
--input_ryan_dir_name="${ENSEMBLE_DIR_NAME}/real_time_predictions/isotonic_regression/for_zhixing/${FAKE_CYCLONE_ID_STRING}" \
--input_zhixing_dir_name="/mnt/shnas10/users/zxruan/nrt_geocenter_g18/preds/best_estimate_center/${FAKE_CYCLONE_ID_STRING}" \
--cyclone_id_string="${CYCLONE_ID_STRING}" \
--input_raw_best_track_file_name="/mnt/shceph/data/ATCF/atcf_rt/ATCF/dat/NHC/b${FAKE_CYCLONE_ID_STRING}.dat" \
--output_file_name="${ENSEMBLE_DIR_NAME}/real_time_predictions/isotonic_regression/comparisons_with_zhixing/${CYCLONE_ID_STRING}.jpg"
