import tarfile
import urllib
import urllib.request


url = "https://chemdata.nist.gov/download/peptide_library/libraries/human/HCD/2020_05_19/human_hcd_tryp_best.msp.tar.gz"
library_file = "human_hcd_tryp_best.msp"

# Download file
_ = urllib.request.urlretrieve(url, f"{library_file}.tar.gz")

# # Extract
with tarfile.open(f"{library_file}.tar.gz") as f:
    f.extractall(".")


with open(library_file, "rt") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 10:
            break

from rich import print, progress  # Rich is a pretty cool library. Google it ;)
import numpy as np
import pandas as pd

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

# break allows us to only stop after the first spectrum is defined
for spectrum in read_msp("human_hcd_tryp_best.msp"):
    print(spectrum["Name"])
    break
