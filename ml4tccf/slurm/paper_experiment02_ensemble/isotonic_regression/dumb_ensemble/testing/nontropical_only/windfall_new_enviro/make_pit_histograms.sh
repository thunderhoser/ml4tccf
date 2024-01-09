#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

python3 -u "${CODE_DIR_NAME}/make_pit_histograms.py" \
--input_prediction_file_pattern="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/predictions*.nc" \
--num_bins=50 \
--output_file_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/pit_histograms.nc"
