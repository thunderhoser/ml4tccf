#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/waf2024"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_best_track/simplified_700x700"
SATELLITE_NORM_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/ir_satellite_normalization_params.zarr"

ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

log_file_name="plot_predictions_2024AL12_first.out"

python3 -u "${CODE_DIR_NAME}/waf2024_plot_predictions.py" &> ${log_file_name} \
--input_prediction_file_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_best_track/isotonic_regression/2024AL12_20241006.nc" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${SATELLITE_NORM_FILE_NAME}" \
--target_time_strings "2024-10-06-1810" "2024-10-06-1710" "2024-10-06-1740" "2024-10-06-1840" "2024-10-06-1910" \
--num_samples_per_target_time=1 \
--lag_times_minutes 0 \
--wavelengths_microns 3.9  6.185  6.95  7.34  8.5  9.61  10.35  11.2  12.3  13.3 \
--prediction_plotting_format_string="probability_contours" \
--prob_colour_map_name="BuGn" \
--prob_contour_smoothing_radius_px=2 \
--prob_contour_opacity=0.7 \
--output_dir_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_best_track/isotonic_regression/2024AL12_20241006"
