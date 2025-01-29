
## Imports
import tarfile
import urllib
import re

import pandas as pd
import numpy as np

from rich import print, progress

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Function to read the MSP format dataset

def read_msp(filename):
    """Iterate over MSP spectral library file and return spectra as dicts."""
    spectrum = {}
    mz = []
    intensity = []
    annotation = []

    with progress.open(filename, "rt") as f:
        for line in f:
            # `Name: ` is the first line of a new entry in the file
            if line.startswith("Name: "):
                if spectrum:
                    # Finalize and yield previous spectrum
                    spectrum["sequence"] = spectrum["Fullname"].split(".")[1]  # Remove the previous/next amino acids
                    spectrum["mz"] = np.array(mz, dtype="float32")
                    spectrum["intensity"] = np.array(intensity, dtype="float32")
                    spectrum["annotation"] = np.array(annotation, dtype="str")
                    yield spectrum

                    # Define new spectrum
                    spectrum = {}
                    mz = []
                    intensity = []
                    annotation = []

                # Extract everything after `Name: `
                spectrum["Name"] = line.strip()[6:]

            elif line.startswith("Comment: "):
                # Parse all comment items as metadata
                metadata = [i.split("=") for i in line[9:].split(" ")]
                for item in metadata:
                    if len(item) == 2:
                        spectrum[item[0]] = item[1]

            elif line.startswith("Num peaks: "):
                spectrum["Num peaks"] = int(line.strip()[11:])

            elif len(line.split("\t")) == 3:
                # Parse peak list items one-by-one
                line = line.strip().split("\t")
                mz.append(line[0])
                intensity.append(line[1])
                annotation.append(line[2].strip('"'))

    # Final spectrum
    spectrum["sequence"] = spectrum["Fullname"].split(".")[1]  # Remove the previous/next amino acids
    spectrum["mz"] = np.array(mz, dtype="float32")
    spectrum["intensity"] = np.array(intensity, dtype="float32")
    spectrum["annotation"] = np.array(annotation, dtype="str")
    yield spectrum

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Functions for data preprocessing (Preparing the spectra for training)

# Normalize the intensities
## TIC Normalization
def tic_normalize(msp_spectrum):
    tic = np.sum(msp_spectrum['intensity'])
    msp_spectrum['intensity'] = msp_spectrum['intensity'] / tic

# Transform the intensities
## Square root transform
def sqrt_transform(msp_spectrum):
    msp_spectrum['intensity'] = np.sqrt(msp_spectrum['intensity'])

# Annotate and parse the peaks intensities to an format suitable for machine learning
## Using regex

def filter_peaks(msp_spectrum):
    """Filter spectrum peaks to only charge 1 b- and y ions"""
    # Generate boolean mask
    get_mask = np.vectorize(lambda x: bool(re.match("^(b|y)([0-9]+)\/" , x)))
    mask = get_mask(msp_spectrum['annotation'])

    msp_spectrum['annotation'] = msp_spectrum['annotation'][mask]
    msp_spectrum['mz'] = msp_spectrum['mz'][mask]
    msp_spectrum['intensity'] = msp_spectrum['intensity'][mask]

def parse_peaks(msp_spectrum, ion_type):

    # Generate vectorized functions
    get_ions = np.vectorize(lambda x : bool(re.match(f"^({ion_type})([0-9]+)\/", x)))
    get_ions_order = np.vectorize(lambda x : re.match(f"^({ion_type})([0-9]+)\/", x)[2])
    
    # Get mask with requested ion types
    mask = get_ions(msp_spectrum["annotation"])
    
    # Create empty array with all possible ions
    n_ions = len(msp_spectrum["sequence"]) - 1
    parsed_intensity = np.zeros(n_ions)
    
    # Check if any ions of this type are present
    if mask.any():
        # Filter for ion type and sort
        ion_order = get_ions_order(msp_spectrum["annotation"][mask]).astype(int) - 1
        # Add ions to correct position in the array
        parsed_intensity[ion_order] = msp_spectrum["intensity"][mask]
        
    # Error check
    try:
        msp_spectrum["parsed_intensity"][ion_type] = parsed_intensity
    except KeyError:
        msp_spectrum["parsed_intensity"] = {}
        msp_spectrum["parsed_intensity"][ion_type] = parsed_intensity

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Parsing the full spectrum library
spectrum_list = []
for msp_spectrum in read_msp("human_hcd_tryp_best.msp"):
    # Process intensities
    tic_normalize(msp_spectrum)
    sqrt_transform(msp_spectrum)
    parse_peaks(msp_spectrum, "b")  # Adds `parsed_intensity` > `b`
    parse_peaks(msp_spectrum, "y")  # Adds `parsed_intensity` > `y`

    # Parse metadata
    spectrum = {
        "sequence": msp_spectrum["sequence"],
        "modifications": msp_spectrum["Mods"],
        "charge": int(msp_spectrum["Charge"]),
        "nce": float(msp_spectrum["NCE"]),
        "parsed_intensity": msp_spectrum["parsed_intensity"]
    }

    # Append to list
    spectrum_list.append(spectrum)

spectrum_df = pd.DataFrame(spectrum_list)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

print(spectrum_df)
