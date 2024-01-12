#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_standalone/ml4tccf"
INPUT_DIR_NAME="/scratch1/BMC/gsd-hpcs/Ryan.Lagerquist/raw_tccf_data"
OUTPUT_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/satellite_data/wp_fuckery_test/processed"

python3 -u "${CODE_DIR_NAME}/process_satellite_data.py" \
--input_dir_name="${INPUT_DIR_NAME}" \
--cyclone_id_string="2020WP16" \
--max_bad_pixels_low_res=2000000 \
--max_bad_pixels_high_res=50000000 \
--temporary_dir_name="${OUTPUT_DIR_NAME}/temporary_2020WP16" \
--min_visible_pixel_fraction=0.9 \
--altitude_angle_exe_name="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4tccf_project/solarpos/solarpos" \
--output_dir_name="${OUTPUT_DIR_NAME}"
