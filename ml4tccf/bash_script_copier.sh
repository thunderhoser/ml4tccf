#!/bin/bash

NEW_CYCLONE_ID_STRINGS=("2024AL16" "2024AL17" "2024AL18" "2024AL19" "2024AL94" "2024AL95" "2024AL97" "2024EP12" "2024EP13" "2024EP14" "2024EP91" "2024EP92" "2024EP93" "2024WP22" "2024WP23" "2024WP24" "2024WP25" "2024WP26" "2024WP27" "2024WP28" "2024WP92" "2024WP94")
TEMPLATE_FILE_NAME="normalize_2024AL09.sh"

for this_id_string in "${NEW_CYCLONE_ID_STRINGS[@]}"; do
    new_file_name="normalize_${this_id_string}.sh"
    cp "$TEMPLATE_FILE_NAME" "$new_file_name"
    sed -i "s/2024AL09/${this_id_string}/g" "$new_file_name"
    echo "Created and updated: $new_file_name"
done
