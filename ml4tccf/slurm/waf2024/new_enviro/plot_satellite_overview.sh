#!/bin/sh

set CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
set SATELLITE_DIR_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalized_for_paper"
set NORMALIZATION_FILE_NAME="/scratch2/BMC/gsd-hpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/processed/normalization_params_for_paper.zarr"
set OUTPUT_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/waf2024/satellite_overview"

python3 -u "${CODE_DIR_NAME}/plot_satellite.py" \
--input_satellite_dir_name="${SATELLITE_DIR_NAME}" \
--input_normalization_file_name="${NORMALIZATION_FILE_NAME}" \
--cyclone_id_string="2020AL29" \
--valid_time_strings "2020-11-09-0400" \
--plot_latlng_coords=1 \
--low_res_wavelengths_microns 3.9  6.185  6.95  7.34  8.5  9.61  10.35  11.2  12.3  13.3 \
--high_res_wavelengths_microns -1 \
--output_dir_name="${OUTPUT_DIR_NAME}/2020AL29"
