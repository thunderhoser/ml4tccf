#!/bin/bash

for this_script_name in process*sh; do
    nohup bash ${this_script_name} &
done
