#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

python3 -u "${CODE_DIR_NAME}/plot_scalar_evaluation.py" \
--input_evaluation_file_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/evaluation.nc" \
--confidence_level=0.95 \
--output_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/testing_by_storm_type/storm-type=non_tropical/evaluation"
