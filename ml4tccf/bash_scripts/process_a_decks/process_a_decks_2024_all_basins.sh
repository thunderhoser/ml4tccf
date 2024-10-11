#!/bin/bash

conda activate base

CODE_DIR_NAME="/home/lagerquist/ml4tccf/ml4tccf/scripts"

LOG_FILE_NAME="process_a_decks_2024_all_basins.out"

python3 -u "${CODE_DIR_NAME}/process_a_deck_data.py" &> ${LOG_FILE_NAME} \
--input_dir_name="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/raw_nhc_and_jtwc" \
--first_year=2024 \
--last_year=2024 \
--output_file_name="/mnt/shnas10/users/lagerquist/ml4tccf_project/a_decks/processed/a_decks_2024_unnormalized.nc"
