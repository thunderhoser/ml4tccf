#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_experiment02_ensemble/tropical_and_nontropical"

model_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble"
echo $model_dir_name

python3 -u "${CODE_DIR_NAME}/plot_evaluation_by_latlng.py" \
--input_evaluation_dir_name="${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_latlng" \
--latitude_spacing_deg=10 \
--longitude_spacing_deg=10 \
--main_font_size=40 \
--label_font_size=15 \
--output_dir_name="${model_dir_name}/isotonic_regression/testing_by_storm_type/storm-type=tropical/testing_by_latlng"
