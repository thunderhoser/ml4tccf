#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/plot_error_histograms_scalar.py" \
--input_prediction_file_pattern="${model_dir_name}/isotonic_regression/testing/predictions_*.nc" \
--plot_orig_errors=1 \
--output_dir_name="${model_dir_name}/isotonic_regression/testing/error_histograms"
