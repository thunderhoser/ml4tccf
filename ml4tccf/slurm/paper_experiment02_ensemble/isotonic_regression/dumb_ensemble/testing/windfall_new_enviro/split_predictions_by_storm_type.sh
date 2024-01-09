#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"
A_DECK_FILE_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/a_decks/including_western_pacific_a-decks_not-b-decks/processed_a_decks_2016-2022.nc"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/split_predictions_by_storm_type.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/testing/predictions_*.nc" \
--input_a_deck_file_name="${A_DECK_FILE_NAME}" \
--make_storm_types_binary=1 \
--output_prediction_dir_name="${model_dir_name}/isotonic_regression/testing_by_storm_type"
