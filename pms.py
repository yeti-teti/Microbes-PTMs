## Imports
import tarfile
import urllib
import re

from rich import print, progress

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from hyperopt import fmin, hp, tpe, STATUS_OK

np.random.seed(42)

library_file = "human_hcd_tryp_best.msp"

# ## Downloading the dataset
# url = "https://chemdata.nist.gov/download/peptide_library/libraries/human/HCD/2020_05_19/human_hcd_tryp_best.msp.tar.gz"

# # Download file
# _ = urllib.request.urlretrieve(url, f"{library_file}.tar.gz")

# with tarfile.open(f"{library_file}.tar.gz") as f:
#     f.extractall("./datasets/")

dataset_location = "./datasets/" + f"{library_file}"
dataset_location

## First 10 lines of the dataset
with open(dataset_location, "rt") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 10:
            break


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

for spectrum in read_msp("human_hcd_tryp_best.msp"):
    print(spectrum["Name"])
    break

pd.DataFrame({
    "mz": spectrum["mz"],
    "intensity": spectrum["intensity"],
    "annotation": spectrum["annotation"]
})


plt.figure(figsize=(10,5))

sup.spectrum(
    sus.MsmsSpectrum(
        identifier = spectrum['Name'],
        precursor_mz = float(spectrum['Parent']),
        precursor_charge = int(spectrum['Charge']),
        mz = spectrum['mz'],
        intensity = spectrum['intensity']
    )
)
plt.title(spectrum['Name'])
plt.show()


# 1. Normalize the intensities
## TIC Normalization
def tic_normalize(msp_spectrum):
    tic = np.sum(msp_spectrum['intensity'])
    msp_spectrum['intensity'] = msp_spectrum['intensity'] / tic

# Before normalization
print(f"Before Normalization: {spectrum['intensity'][:10]}")
# After normalization
tic_normalize(spectrum)
print(f"After Normalization: {spectrum['intensity'][:10]}")


# 2. Transform the intensities
## Square root transform

sns.set_style("whitegrid")

# Before transform
sns.displot(spectrum['intensity'], bins=20)
plt.show()

# After transform
def sqrt_transform(msp_spectrum):
    msp_spectrum['intensity'] = np.sqrt(msp_spectrum['intensity'])

# After transform
sqrt_transform(spectrum)
sns.displot(spectrum['intensity'], bins=20)
plt.show()


# 3. Annotate the peak
plt.figure(figsize=(12, 5))
sup.spectrum(
    sus.MsmsSpectrum(
        identifier = spectrum['Name'], 
        precursor_mz = float(spectrum['Parent']),
        precursor_charge = int(spectrum['Charge']), 
        mz = spectrum['mz'],
        intensity = spectrum['intensity'],
        peptide = spectrum['sequence']
    ).annotate_peptide_fragments(25, 'ppm')
)
plt.title(spectrum["Name"])
plt.show()


# 4. Parse the relevant peak intensities to an format suitable for machine learning
## Using regex

def filter_peaks(msp_spectrum):
    """Filter spectrum peaks to only charge 1 b- and y ions"""
    # Generate boolean mask
    get_mask = np.vectorize(lambda x: bool(re.match("^(b|y)([0-9]+)\/" , x)))
    mask = get_mask(msp_spectrum['annotation'])

    msp_spectrum['annotation'] = msp_spectrum['annotation'][mask]
    msp_spectrum['mz'] = msp_spectrum['mz'][mask]
    msp_spectrum['intensity'] = msp_spectrum['intensity'][mask]

filter_peaks(spectrum)

plt.figure(figsize=(12, 5))
sup.spectrum(
    sus.MsmsSpectrum(
        identifier = spectrum['Name'], 
        precursor_mz = float(spectrum['Parent']),
        precursor_charge = int(spectrum['Charge']), 
        mz = spectrum['mz'],
        intensity = spectrum['intensity'],
        peptide = spectrum['sequence']
    ).annotate_peptide_fragments(25, 'ppm')
)
plt.title(spectrum["Name"])
plt.show()

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

parse_peaks(spectrum, "b")
parse_peaks(spectrum, "y")

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
spectrum_df

# Train / Validation / Test split

train_val_peptides, test_peptides = train_test_split(spectrum_df["sequence"].unique(), train_size=0.9)
train_val_spectra = spectrum_df[spectrum_df["sequence"].isin(train_val_peptides)]
test_spectra = spectrum_df[spectrum_df["sequence"].isin(test_peptides)]

train_val_spectra.reset_index().to_feather("fragmentation-nist-humanhcd20160503-parsed-trainval.feather")
test_spectra.reset_index().to_feather("fragmentation-nist-humanhcd20160503-parsed-test.feather")


# Data preparation

# A. Feature Engineering
# 1. Reading the parsed spectral library
train_val_spectra = pd.read_feather("fragmentation-nist-humanhcd20160503-parsed-trainval.feather")
test_spectra = pd.read_feather("fragmentation-nist-humanhcd20160503-parsed-test.feather")

# 2. Feature engineering
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
properties = np.array([
    [37,35,59,129,94,0,210,81,191,81,106,101,117,115,343,49,90,60,134,104],  # basicity
    [68,23,33,29,70,58,41,73,32,73,66,38,0,40,39,44,53,71,51,55],  # helicity
    [51,75,25,35,100,16,3,94,0,94,82,12,0,22,22,21,39,80,98,70],  # hydrophobicity
    [32,23,0,4,27,32,48,32,69,32,29,26,35,28,79,29,28,31,31,28],  # pI
])

pd.DataFrame(properties, columns=amino_acids, index=["basicity", "helicity", "hydrophobicity", "pI"])

def encode_peptide(sequence, charge):
    # 4 properties * 5 quantiles * 3 ion types + 4 properties * 4 site + 2 global
    n_features = 78
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    n_ions = len(sequence) - 1

    # Encode amino acids as integers to index amino acid properties for peptide sequence
    aa_indices = {aa: i for i, aa in  enumerate("ACDEFGHIKLMNPQRSTVWY")}
    aa_to_index = np.vectorize(lambda aa: aa_indices[aa])
    peptide_indexed = aa_to_index(np.array(list(sequence)))
    peptide_properties = properties[:, peptide_indexed]

    # Empty peptide_features array
    peptide_features = np.full((n_ions, n_features), np.nan)

    for b_ion_number in range(1, n_ions + 1):
        # Calculate quantiles of features across peptide, b-ion, and y-ion
        peptide_quantiles = np.hstack(
            np.quantile(peptide_properties, quantiles, axis=1).transpose()
        )
        b_ion_quantiles = np.hstack(
            np.quantile(peptide_properties[:,:b_ion_number], quantiles, axis=1).transpose()
        )
        y_ion_quantiles = np.hstack(
            np.quantile(peptide_properties[:,b_ion_number:], quantiles, axis=1).transpose()
        )

        # Properties on specific sites: nterm, frag-1, frag+1, cterm
        specific_site_indexes = np.array([0, b_ion_number - 1, b_ion_number, -1])
        specific_site_properties = np.hstack(peptide_properties[:, specific_site_indexes].transpose())

        # Global features: Length and charge
        global_features = np.array([len(sequence), int(charge)])

        # Assign to peptide_features array
        peptide_features[b_ion_number - 1, 0:20] = peptide_quantiles
        peptide_features[b_ion_number - 1, 20:40] = b_ion_quantiles
        peptide_features[b_ion_number - 1, 40:60] = y_ion_quantiles
        peptide_features[b_ion_number - 1, 60:76] = specific_site_properties
        peptide_features[b_ion_number - 1, 76:78] = global_features

    return peptide_features


def generate_feature_names():
    feature_names = []
    for level in ["peptide", "b", "y"]:
        for aa_property in ["basicity", "helicity", "hydrophobicity", "pi"]:
            for quantile in ["min", "q1", "q2", "q3", "max"]:
                feature_names.append("_".join([level, aa_property, quantile]))
    for site in ["nterm", "fragmin1", "fragplus1", "cterm"]:
        for aa_property in ["basicity", "helicity", "hydrophobicity", "pi"]:
            feature_names.append("_".join([site, aa_property]))

    feature_names.extend(["length", "charge"])
    return feature_names

# B. Getting the target intensities
test_spectrum = train_val_spectra.iloc[4]
test_spectrum

peptide_targets = pd.DataFrame({
    "b_target": test_spectrum["parsed_intensity"]["b"],
    "y_target": test_spectrum["parsed_intensity"]["y"]
})
peptide_targets

peptide_targets =  pd.DataFrame({
    "b_target": test_spectrum["parsed_intensity"]["b"],
    "y_target": test_spectrum["parsed_intensity"]["y"][::-1],
})
peptide_targets

features = encode_peptide(test_spectrum["sequence"], test_spectrum["charge"])
targets = np.stack([test_spectrum["parsed_intensity"]["b"], test_spectrum["parsed_intensity"]["y"][::-1]], axis=1)
spectrum_id = np.full(shape=(targets.shape[0], 1), fill_value=test_spectrum["index"])  # Repeat id for all ions

def generate_ml_input(spectra):
    tables = []
    for spectrum in progress.track(spectra.to_dict(orient="records")):
        features = encode_peptide(spectrum["sequence"], spectrum["charge"])
        targets = np.stack([spectrum["parsed_intensity"]["b"], spectrum["parsed_intensity"]["y"][::-1]], axis=1)
        spectrum_id = np.full(shape=(targets.shape[0], 1), fill_value=spectrum["index"])  # Repeat id for all ions
        table = np.hstack([spectrum_id, features, targets])
        tables.append(table)

    full_table = np.vstack(tables)
    spectra_encoded = pd.DataFrame(full_table, columns=["spectrum_id"] + generate_feature_names() + ["b_target",  "y_target"])
    return spectra_encoded

train_val_encoded = generate_ml_input(train_val_spectra)
train_val_encoded.to_feather("fragmentation-nist-humanhcd20160503-parsed-trainval-encoded.feather")

test_encoded = generate_ml_input(test_spectra)
test_encoded.to_feather("fragmentation-nist-humanhcd20160503-parsed-test-encoded.feather")

# 3. Training the model
reg =  GradientBoostingRegressor()

X_train = train_val_encoded.drop(columns=["spectrum_id", "b_target",  "y_target"])
y_train = train_val_encoded["y_target"]
X_test = test_encoded.drop(columns=["spectrum_id", "b_target",  "y_target"])
y_test = test_encoded["y_target"]

reg.fit(X_train, y_train)

y_test_pred = reg.predict(X_test)
np.corrcoef(y_test, y_test_pred)[0][1]

# Hyperparameter optimization
def objective(n_estimators):
    # Define algorithm
    reg =  GradientBoostingRegressor(n_estimators=n_estimators)

    # Fit model
    reg.fit(X_train, y_train)

    # Test model
    y_test_pred = reg.predict(X_test)
    correlation = np.corrcoef(y_test, y_test_pred)[0][1]

    return {'loss': -correlation, 'status': STATUS_OK} 


reg =  GradientBoostingRegressor(n_estimators=946)

X_train = train_val_encoded.drop(columns=["spectrum_id", "b_target",  "y_target"])
y_train = train_val_encoded["y_target"]
X_test = test_encoded.drop(columns=["spectrum_id", "b_target",  "y_target"])
y_test = test_encoded["y_target"]

reg.fit(X_train, y_train)

y_test_pred = reg.predict(X_test)

np.corrcoef(y_test, y_test_pred)[0][1]


prediction_df_y = pd.DataFrame({
    "spectrum_id": test_encoded['spectrum_id'] ,
    "target_y": y_test,
    "prediction_y": y_test_pred
})

corr_y = prediction_df_y.groupby("spectrum_id").corr().iloc[::2]['prediction_y']
corr_y.index = corr_y.index.droplevel(1)
corr_y = corr_y.reset_index().rename(columns={"prediction_y": "correlation"})
corr_y

sns.catplot(
    data=corr_y, x="correlation", 
    fliersize=1, 
    kind = 'box', aspect=4, height=2
)
plt.show()