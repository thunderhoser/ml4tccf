#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/simplified_700x700"
SATELLITE_NORM_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/ir_satellite_normalization_params.zarr"

ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

log_file_name="plot_predictions_2024AL14_2020-10-09-0130.sh_${CYCLONE_ID_STRING}_${valid_date_string}.out"

python3 -u "${CODE_DIR_NAME}/waf2024_plot_predictions_multi_aug.py" &> ${log_file_name} \
--input_prediction_file_name="${ENSEMBLE_DIR_NAME}/predictions/isotonic_regression/2024AL14_20241009.nc" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${SATELLITE_NORM_FILE_NAME}" \
--target_time_strings "2024-10-09-0130" \
--wavelength_microns=10.350 \
--prediction_plotting_format_string="probability_contours" \
--prob_colour_map_name="BuGn" \
--prob_contour_smoothing_radius_px=2 \
--output_dir_name="${ENSEMBLE_DIR_NAME}/predictions/isotonic_regression/2024AL14_20241009"
