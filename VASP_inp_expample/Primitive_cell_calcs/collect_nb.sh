#!/bin/bash

# Output file to store combined content
output_file="niobium_primitive.xyz"

# Clear the output file if it already exists
> "$output_file"

# Loop through all Ta_pv_prim_* folders
for folder in Nb_prim_*/; do
    # Check if the folder contains primitive.xyz
    if [[ -f "${folder}primitive.xyz" ]]; then
        # Append the content of primitive.xyz to the output file
	echo "processing ${folder}"
        cat "${folder}primitive.xyz" >> "$output_file"
    else
        echo "Warning: ${folder} does not contain primitive.xyz"
    fi
done

echo "All primitive.xyz files have been combined into $output_file"
