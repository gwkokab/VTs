import h5py
import numpy as np

# Set input and output filenames
input_file = "o1+o2+o3_imbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
output_file = "filtered_output.hdf5"

# Define filtering function (change this as needed)
def filter_condition(data_dict):
    return data_dict["mass1_source"] < 100  # Example condition

# Load input file
with h5py.File(input_file, "r") as infile:
    injections_group = infile["/injections"]
    
    # Load all datasets into memory
    all_data = {key: injections_group[key][()] for key in injections_group}

    # Apply the condition
    mask = filter_condition(all_data)
    
    # Filter all datasets with the same mask
    filtered_data = {key: value[mask] for key, value in all_data.items()}

    # Write to output file
    with h5py.File(output_file, "w") as outfile:
        outgroup = outfile.create_group("injections")
        for key, value in filtered_data.items():
            outgroup.create_dataset(key, data=value)

print(f"Filtered file saved to: {output_file}")
