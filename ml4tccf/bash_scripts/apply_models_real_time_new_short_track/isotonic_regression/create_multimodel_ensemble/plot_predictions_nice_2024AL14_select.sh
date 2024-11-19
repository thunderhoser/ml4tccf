#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/waf2024"
SATELLITE_DIR_NAME="/mnt/nvme-data5/lagerquist/ml4tccf_project/satellite_data/processed/normalized_params_for_paper/recentered_on_short_track/simplified_700x700"
SATELLITE_NORM_FILE_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/ir_satellite_normalization_params.zarr"

CYCLONE_ID_STRING="2024AL14"
ENSEMBLE_DIR_NAME="/mnt/shnas10/users/lagerquist/ml4tccf_project/geocenter_models/ensemble"

log_file_name="plot_predictions_nice_${CYCLONE_ID_STRING}_select.out"

python3 -u "${CODE_DIR_NAME}/wpo2025_plot_predictions.py" &> ${log_file_name} \
--input_prediction_file_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/${CYCLONE_ID_STRING}_20241007.nc" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${SATELLITE_NORM_FILE_NAME}" \
--input_best_track_dir_name="/mnt/shnas10/users/galka/track_interpolated/best_track/bal142024" \
--target_time_strings "2024-10-07-2140" \
--lag_times_minutes 0 \
--wavelengths_microns 3.9 6.95 10.35 \
--prediction_plotting_format_string="probability_contours" \
--prob_colour_map_name="BuGn" \
--prob_contour_smoothing_radius_px=2 \
--prob_contour_opacity=0.7 \
--output_dir_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/${CYCLONE_ID_STRING}_select"

python3 -u "${CODE_DIR_NAME}/wpo2025_plot_predictions.py" &>> ${log_file_name} \
--input_prediction_file_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/${CYCLONE_ID_STRING}_20241008.nc" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${SATELLITE_NORM_FILE_NAME}" \
--input_best_track_dir_name="/mnt/shnas10/users/galka/track_interpolated/best_track/bal142024" \
--target_time_strings "2024-10-08-2350" \
--lag_times_minutes 0 \
--wavelengths_microns 10.35 \
--prediction_plotting_format_string="probability_contours" \
--prob_colour_map_name="BuGn" \
--prob_contour_smoothing_radius_px=2 \
--prob_contour_opacity=0.7 \
--output_dir_name="${ENSEMBLE_DIR_NAME}/real_time_predictions_new_short_track/isotonic_regression/${CYCLONE_ID_STRING}_select"
