#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/plot_evaluation_by_basin.py" \
--input_evaluation_file_names "${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_basin/basin=AL/evaluation.nc" "${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_basin/basin=EP/evaluation.nc" "${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_basin/basin=WP/evaluation.nc" \
--confidence_level=0.95 \
--output_dir_name="${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_basin"
