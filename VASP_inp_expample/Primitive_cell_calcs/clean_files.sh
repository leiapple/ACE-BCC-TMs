#!/bin/bash

# List of files to delete
files_to_delete=("EIGENVAL" "CHG" "CHGCAR" "PROCAR" "POTCAR" "IBZKPT" "ase-sort.dat" "vasprun.xml")

# Find all directories (including nested ones) and loop through them
find . -type d | while read -r dir; do
    echo "Checking directory: $dir"
    
    # Loop through each file in the list
    for file in "${files_to_delete[@]}"; do
        # Check if the file exists in the directory
        if [ -f "$dir/$file" ]; then
            # If the file exists, delete it
            echo "Deleting $file in $dir"
            rm "$dir/$file"
        else
            # If the file does not exist, print a message
            echo "$file not found in $dir"
        fi
    done
done
