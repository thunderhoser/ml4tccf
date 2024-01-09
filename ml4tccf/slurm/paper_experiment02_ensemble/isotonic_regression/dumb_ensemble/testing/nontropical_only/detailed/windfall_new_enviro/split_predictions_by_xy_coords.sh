#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"
EXTENDED_BEST_TRACK_FILE_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/extended_best_track_data_including2022.nc"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/split_predictions_by_xy_coords.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/predictions_*.nc" \
--input_extended_best_track_file_name="${EXTENDED_BEST_TRACK_FILE_NAME}" \
--x_coord_cutoffs_metres -3000000 -1500000 0 1500000 3000000 4500000 \
--y_coord_cutoffs_metres 1500000 3000000 4500000 \
--output_prediction_dir_name="${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/testing_by_xy_coords"
