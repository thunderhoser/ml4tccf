#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

python3 -u "${CODE_DIR_NAME}/run_discard_test.py" \
--input_prediction_file_pattern="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/predictions*.nc" \
--discard_fractions 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
--output_file_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/discard_test.nc"
