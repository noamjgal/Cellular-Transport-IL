#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:16:17 2024

@author: noamgal
"""

import csv
import shutil
import tempfile
import os

def update_translations(input_file):
    """Update the translations.txt file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as infile, temp_file:
        reader = csv.reader(infile)
        writer = csv.writer(temp_file)
        
        # Write new header
        writer.writerow(['trans_id', 'table_name', 'field_name', 'language', 'translation'])
        
        # Skip the old header
        next(reader)
        
        # Process and write each row
        for row in reader:
            if len(row) == 5 and all(field.strip() for field in row):
                trans_id, table_name, _, language, translation = [field.strip() for field in row]
                writer.writerow([trans_id, table_name, 'name', language, translation])

    replace_file(temp_file.name, input_file)

def update_routes(input_file):
    """Update the routes.txt file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', encoding='utf-8')

    def map_route_type(route_type):
        return '3' if route_type == '8' else '2' if route_type == '715' else route_type

    with open(input_file, 'r', encoding='utf-8') as infile, temp_file:
        reader = csv.reader(infile)
        writer = csv.writer(temp_file)
        
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            if len(row) > 5:
                row[5] = map_route_type(row[5])
            writer.writerow(row)

    replace_file(temp_file.name, input_file)

def update_stops(input_file):
    """Update the stops.txt file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', encoding='utf-8')

    def map_location_type(loc_type):
        return '0' if loc_type in ['1', '2', '3', '4'] else loc_type

    with open(input_file, 'r', encoding='utf-8') as infile, temp_file:
        reader = csv.reader(infile)
        writer = csv.writer(temp_file)
        
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            if len(row) > 6:
                row[6] = map_location_type(row[6])
            writer.writerow(row)

    replace_file(temp_file.name, input_file)

def replace_file(temp_file, original_file):
    """Replace the original file with the temporary file."""
    shutil.move(temp_file, original_file)
    print(f"Updated {original_file}")

def clean_gtfs(directory):
    """Clean GTFS files in the specified directory."""
    files_to_clean = {
        'translations.txt': update_translations,
        'routes.txt': update_routes,
        'stops.txt': update_stops
    }

    for filename, update_function in files_to_clean.items():
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            update_function(file_path)
        else:
            print(f"Warning: {filename} not found in the specified directory.")

if __name__ == "__main__":
    # Replace with the path to your Israel Public Transportation GTFS directory
    gtfs_directory = '/path/to/your/gtfs/directory'
    clean_gtfs(gtfs_directory)