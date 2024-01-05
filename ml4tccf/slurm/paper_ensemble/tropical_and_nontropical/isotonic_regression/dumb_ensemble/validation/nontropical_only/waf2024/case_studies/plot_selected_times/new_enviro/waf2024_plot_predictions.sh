#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
# SATELLITE_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_for_paper/simplified_all_wavelengths"
SATELLITE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_for_paper/800x800_grids"
NORMALIZATION_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalization_params_for_paper.zarr"
TOP_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_models/paper_ensemble/tropical_and_nontropical"

cyclone_id_string=$1
target_times_one_string=$2

python3 -u "${CODE_DIR_NAME}/waf2024_plot_predictions.py" \
--input_prediction_file_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/validation_by_storm_type/storm-type=non_tropical/predictions_${cyclone_id_string}.nc" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${NORMALIZATION_FILE_NAME}" \
--target_time_strings ${target_times_one_string} \
--lag_times_minutes 0 \
--wavelengths_microns 3.9  6.185  6.95  7.34  8.5  9.61  10.35  11.2  12.3  13.3 \
--prediction_plotting_format_string="probability_contours" \
--prob_colour_map_name="BuGn" \
--prob_contour_smoothing_radius_px=5 \
--point_prediction_marker_size=36 \
--output_dir_name="${TOP_MODEL_DIR_NAME}/dumb_ensemble/isotonic_regression/validation_by_storm_type/storm-type=non_tropical/waf2024/case_study_${cyclone_id_string}/selected_times"
