# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: stdprops
#     language: python
#     name: python3
# ---

# # NIST NBS tables exploratory data analysis
#
# The goal of this notebook is to explore, check, and clear the publicly available [NIST NBS Tables](https://data.nist.gov/od/id/mds2-2124) in order to understand and provide insights about the data. This notebook also prepares the data for further developments, like machine learning std properties estimation.

# ## Data Preprocessing

# +
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio

from pathlib import Path
import os

DATA_PATH = Path(os.environ.get("DATAPATH"))

pio.renderers.default = "png"
# -

# Reading the raw data:

# +
df_nist_table_raw = pd.read_csv(DATA_PATH / "NBS_Tables.csv", skiprows=1)

df_nist_table_raw
# -

# Convenient columns renaming:

# +
columns_to_rename = {
    "State\nDescription": "State Description",
    "Molar Mass\ng mol-1": "Molar Mass",
    "ΔfH°": "deltaH0",
    "ΔfG°\nkJ mol-1": "deltaG0",
    "       S°              \nJ mol-1 K-1": "S0",
    "Cp\nJ mol-1 K-1": "Cp"
}
df_nist_table_cols_renamed = df_nist_table_raw.rename(columns=columns_to_rename)

df_nist_table_cols_renamed
# -

# Select only the columns of interest:

df_nist_table_cols_renamed.columns

# +
target_columns = [
    "Formula", "State", "Molar Mass", "deltaH0", "deltaG0", "S0", "Cp"
]

df_nist_tables_full = df_nist_table_cols_renamed[target_columns].copy()

df_nist_tables_full.head(30)
# -

# Replace `-` fields by NaN:

# +
df_nist_tables_full.replace({r"\*": ""}, regex=True, inplace=True)
df_nist_tables_full.replace("-", np.nan, inplace=True)

df_nist_tables_full.head(30)
# -

df_nist_tables_full["State"].value_counts()

# Now, we can clean the entries that have any NaN since whatever the NaN is, it is not possible to be used in our analysis.

# +
df_nist_tables = df_nist_tables_full.dropna(ignore_index=True)

df_nist_tables.head(30)
# -

# Let's check what left:

df_nist_tables['State'].value_counts()

print(f"Species: {df_nist_tables.shape[0]}\t Fields: {df_nist_tables.shape[1]}")

# Let's now add a new feature to help our analysis (species' eletric charge):

# +
df_nist_tables['Charge'] = df_nist_tables.shape[0] * [0.0]

df_nist_tables
# -

# We need to update the values according to the species' actual charges:

# +
import re

for index, row in df_nist_tables.iterrows():
    species_formula = row["Formula"]
    if "sup" in species_formula:
        numbers_in_formula_name = re.findall(r'\d+', species_formula)
        # Cations
        if "<sup>+" in species_formula:
            df_nist_tables.loc[index, "Charge"] = 1.0
        elif "+" in species_formula:
            df_nist_tables.loc[index, "Charge"] = float(numbers_in_formula_name[-1])
            
        # Anions
        if "<sup>-" in species_formula:
            df_nist_tables.loc[index, "Charge"] = -1.0
        elif "-" in species_formula:
            df_nist_tables.loc[index, "Charge"] = -float(numbers_in_formula_name[-1])
        
        # Uncomment to verify the modifications
        # print(
        #     f"Formula: {row["Formula"]}\t Found numbers: {numbers_in_formula_name}\t Charge: {df_nist_tables.loc[index, "Charge"]}"
        # )
# -

df_nist_tables.head(30)

# Exporting the preprocessed data to a `csv` file to aid further studies:

df_nist_tables.to_csv(DATA_PATH / "NBS_Tables_preprocessed.csv")

# ## Data Exploration

# ### Check correlations
#
# Let's check if the numerical data are correlated for the most present states of matter:

# #### Overall

# +
numerical_columns = [
    "Molar Mass", "deltaH0", "deltaG0", "S0", "Cp", "Charge"
]
df_nist_tables_numerical = df_nist_tables[numerical_columns]
df_nist_tables_numerical = df_nist_tables_numerical.astype(float)
df_nist_tables[numerical_columns] = df_nist_tables[numerical_columns].astype(float)

df_nist_tables_numerical.head(15)
# -

# Molar mass distribution:

fig = px.violin(df_nist_tables, y="Molar Mass", box=True, points='all')
fig.show()

fig = px.histogram(df_nist_tables, x="Molar Mass")
fig.show()

# Molar mass distribution by state of matter:

fig = px.violin(df_nist_tables, x="State", y="Molar Mass", box=True, points='all')
fig.show()

# #### Solid (cr)

# +
df_nist_tables_solids = df_nist_tables[
    df_nist_tables["State"].isin(["cr", "cr2"])
]

df_nist_tables_solids

# +
numerical_columns = [
    "Molar Mass", "deltaH0", "deltaG0", "S0", "Cp", "Charge"
]
df_nist_tables_solids_numerical = df_nist_tables_solids[numerical_columns]
df_nist_tables_solids_numerical = df_nist_tables_solids_numerical.astype(float)

df_nist_tables_solids_numerical.head(15)
# -

df_solids_corr = df_nist_tables_solids_numerical.corr()

fig = px.imshow(df_solids_corr, text_auto=True, aspect="auto", color_continuous_scale='ice', zmin=-2)
fig.show()

# The correlation between the std properties are expected! However, given the molar mass and charge (which does not apply to solids), we want to predict the std properties. Let's continue and check for the other states.

# Checking how `deltaH0` varies with the Molar Mass:

fig = px.scatter(
    df_nist_tables_solids_numerical, x="Molar Mass", y="deltaH0"
)
fig.show()

# Checking how `deltaG0` varies with the Molar Mass:

fig = px.scatter(
    df_nist_tables_solids_numerical, x="Molar Mass", y="deltaG0"
)
fig.show()

# `S0`:

fig = px.scatter(
    df_nist_tables_solids_numerical, x="Molar Mass", y="S0"
)
fig.show()

# `Cp`:

fig = px.scatter(
    df_nist_tables_solids_numerical, x="Molar Mass", y="Cp"
)
fig.show()

# The distribution of molar mass values:

fig = px.histogram(df_nist_tables_solids_numerical, x="Molar Mass")
fig.show()

# #### Gas species

# +
df_nist_tables_gas = df_nist_tables[
    df_nist_tables["State"].isin(["g", "g2"])
]

df_nist_tables_gas_numerical = df_nist_tables_gas[numerical_columns]
df_nist_tables_gas_numerical = df_nist_tables_gas_numerical.astype(float)

df_nist_tables_gas_numerical.head(15)
# -

df_gas_corr = df_nist_tables_gas_numerical.corr()

fig = px.imshow(df_gas_corr, text_auto=True, aspect="auto", color_continuous_scale='ice', zmin=-2)
fig.show()

# Checking how the std properties vary with the molar mass:

# `deltaH0`:

fig = px.scatter(
    df_nist_tables_gas_numerical, x="Molar Mass", y="deltaH0"
)
fig.show()

# `deltaG0`:

fig = px.scatter(
    df_nist_tables_gas_numerical, x="Molar Mass", y="deltaG0"
)
fig.show()

# `S0`:

fig = px.scatter(
    df_nist_tables_gas_numerical, x="Molar Mass", y="S0"
)
fig.show()

# `Cp`:

fig = px.scatter(
    df_nist_tables_gas_numerical, x="Molar Mass", y="Cp"
)
fig.show()

# Check the distribution of molar mass values:

fig = px.histogram(df_nist_tables_gas_numerical, x="Molar Mass")
fig.show()

# #### Aqueous species

# +
df_nist_tables_aq = df_nist_tables[
    df_nist_tables["State"].isin(["ai", "ao"])
]

df_nist_tables_aq_numerical = df_nist_tables_aq[numerical_columns]
df_nist_tables_aq_numerical = df_nist_tables_aq_numerical.astype(float)
# -

df_aq_corr = df_nist_tables_aq_numerical.corr()

fig = px.imshow(df_aq_corr, text_auto=True, aspect="auto", color_continuous_scale='ice', zmin=-2)
fig.show()

# Checking how the std properties vary with the molar mass:

# `deltaH0`:

fig = px.scatter(
    df_nist_tables_aq_numerical, x="Molar Mass", y="deltaH0"
)
fig.show()

# `deltaG0`:

fig = px.scatter(
    df_nist_tables_aq_numerical, x="Molar Mass", y="deltaG0"
)
fig.show()

# `S0`:

fig = px.scatter(
    df_nist_tables_aq_numerical, x="Molar Mass", y="S0"
)
fig.show()

# `Cp`:

fig = px.scatter(
    df_nist_tables_aq_numerical, x="Molar Mass", y="Cp"
)
fig.show()

# Checking how the std properties vary with the charge:

# `deltaH0`:

fig = px.violin(
    df_nist_tables_aq_numerical, x="Charge", y="deltaH0", box=True, points="all"
)
fig.show()

# `deltaG0`:

fig = px.violin(
    df_nist_tables_aq_numerical, x="Charge", y="deltaG0", points="all", box=True
)
fig.show()

# `S0`:

fig = px.violin(
    df_nist_tables_aq_numerical, x="Charge", y="S0", points="all", box=True
)
fig.show()

# `Cp`:

fig = px.violin(
    df_nist_tables_aq_numerical, x="Charge", y="Cp", points="all", box=True
)
fig.show()

# #### Liquid species

# +
df_nist_tables_liq = df_nist_tables[
    df_nist_tables["State"].isin(["l", "l2"])
]

df_nist_tables_liq_numerical = df_nist_tables_liq[numerical_columns]
df_nist_tables_liq_numerical = df_nist_tables_liq_numerical.astype(float)
# -

df_liq_corr = df_nist_tables_liq_numerical.corr()

fig = px.imshow(df_liq_corr, text_auto=True, aspect="auto", color_continuous_scale='ice', zmin=-2)
fig.show()

# Checking how the std properties vary with the molar mass:

# `deltaH0`:

fig = px.scatter(
    df_nist_tables_liq_numerical, x="Molar Mass", y="deltaH0"
)
fig.show()

# `deltaG0`:

fig = px.scatter(
    df_nist_tables_liq_numerical, x="Molar Mass", y="deltaG0"
)
fig.show()

# `S0`:

fig = px.scatter(
    df_nist_tables_liq_numerical, x="Molar Mass", y="S0"
)
fig.show()

# `Cp`:

fig = px.scatter(
    df_nist_tables_liq_numerical, x="Molar Mass", y="Cp"
)
fig.show()



# ### Std thermodynamic properties consistency
#
# The goal here is to perform the G-H-S (standard free Gibbs energy of formation, enthalpy, and entropy). These quantities are related through the formula:
#
# \begin{equation*}
# \Delta_{\mathrm{f}} G^{\circ}=\Delta_{\mathrm{f}} H^{\circ}-T \Delta_{\mathrm{f}} S^{\circ}
# \end{equation*}
#
# With $\Delta_{\mathrm{f}} S^{\circ}$ calculated as
#
# \begin{equation*}
# \Delta_{\mathrm{f}} S^{\circ} = S^{\circ}_{\text{species}} - \sum_{e \in \text{Elements}}S^{\circ}_e
# \end{equation*}
#
# The reference temperature is set as $T = 298.15$ K.
#
# However, to check the consistency we need to retrieve the chemical formulas and the associated elements, and then compute the entropy from the elements. This required and additional database of elements collected from [CHNOSZ](https://github.com/jedick/CHNOSZ/blob/main/inst/extdata/thermo/element.csv). Let's use it as a base to build an equivalent but using NIST/NBS data.

# Reading the CHNOSZ element data:

# +
df_chnosz_elements = pd.read_csv(DATA_PATH / "chnosz_elements (modified).csv")

df_chnosz_elements

# +
map_states_chnosz_to_nist = {
    "gas": "g",
    "liq": "l",
    "aq": "ao"
}

df_chnosz_elements.replace(map_states_chnosz_to_nist, inplace=True)

df_chnosz_elements.head(20)
# -

# Now, we replicate the same structure but collecting the data from NIST/NBS tables:

# +
nist_elements = {
    "element": [],
    "state": [],
    "S0": [],
}

for index, row in df_chnosz_elements.iterrows():
    element_name = row["element"]
    if element_name == "Z":
        continue
    
    element_state = row["state"]
    element_n = row["n"]
    suffix_name = element_n if int(element_n) > 1 else ""
    nist_name = element_name + str(suffix_name)
    
    df_corresponding_nist = df_nist_tables.loc[
        (df_nist_tables["Formula"] == nist_name) & (df_nist_tables["State"] == element_state)
    ]
    
    nist_S0_value = df_corresponding_nist["S0"].values
    element_S0 = nist_S0_value[0] / element_n if len(nist_S0_value) > 0 else None
    
    nist_elements["element"].append(element_name)
    nist_elements["state"].append(element_state)
    nist_elements["S0"].append(element_S0)

df_nist_elements = pd.DataFrame.from_dict(nist_elements)
df_nist_elements.dropna(inplace=True)
df_nist_elements.to_csv(DATA_PATH / "nist_elements.csv")  # check if needed
df_nist_elements


# -

# Define a convenient function to retrieve the amount of each element in a given chemical formula:

def parse_chemical_formula(formula: str) -> dict[str, int]:
    """
    Convenient function to parser and get the amount of elements in
    chemical species formulas.
    """
    import re
    from collections import defaultdict
    
    # Function to parse a molecule or sub-molecule
    def parse_molecule(molecule, multiplier=1):
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', molecule)
        for element, count in elements:
            count = int(count) if count else 1
            element_counts[element] += count * multiplier

    # Remove HTML charge notations
    formula = re.sub(r'<[^>]+>', '', formula)

    # Split the formula into molecules and process each part
    element_counts = defaultdict(int)
    molecules = formula.split('·')
    
    for molecule in molecules:
        # Handle molecules with and without parentheses
        if '(' in molecule:
            while '(' in molecule:
                # Find and replace the innermost parenthetical expression
                sub_molecule, sub_multiplier = re.search(r'\(([A-Za-z0-9]+)\)(\d*)', molecule).groups()
                sub_multiplier = int(sub_multiplier) if sub_multiplier else 1
                molecule = re.sub(r'\(([A-Za-z0-9]+)\)(\d*)', '', molecule, 1)
                parse_molecule(sub_molecule, sub_multiplier)
        
        # Handle preffix-like multiplier
        else:
            sub_multiplier, sub_molecule = re.search(r'(\d*)([A-Za-z0-9]+)', molecule).groups()
            sub_multiplier = int(sub_multiplier) if sub_multiplier else 1
            molecule = re.sub(r'(\d*)([A-Za-z0-9]+)', '', molecule, 1)
            parse_molecule(sub_molecule, sub_multiplier)
            
        # Process the remaining parts of the molecule
        parse_molecule(molecule)

    return dict(element_counts)


# Iterate over all NIST data and compute the GHS residual:

df_nist_tables.head(30)

# +
aqueous_states = [
    "ai", "ao", "aq", "ai2", "ao2", "aq2", "aq3", "ao4", "aq4"
]

H2_species_refdata = df_nist_tables.loc[
    (df_nist_tables["Formula"] == "H2") & (df_nist_tables["State"] == "g")
]

GHS_residuals = []
T = 298.15
for index, row in df_nist_tables.iterrows():
    species_formula = row["Formula"]
    species_state = row["State"]
    is_species_aqueous = species_state in aqueous_states
    elements_in_species = parse_chemical_formula(species_formula)
    
    species_dG0 = row["deltaG0"]
    species_dH0 = row["deltaH0"]
    species_S0 = row["S0"]
    
    elements_S0 = 0.0
    try:
        for element, count in elements_in_species.items():
            df_element = df_nist_elements.loc[df_nist_elements['element'] == element]
            elements_S0 += df_element['S0'].values[0] * count
        
        if is_species_aqueous:
            species_charge = row["Charge"]
            H2_g_S0 = H2_species_refdata['S0'].values[0]
            elements_S0 += (species_charge / 2) * H2_g_S0 * T
    except IndexError:
        print(f"Skipping species {species_formula}: element {element} is lacking")
        elements_S0 = np.nan
    
    species_dS0 = species_S0 - elements_S0
    
    GHS_residual = species_dG0 * 1000 - species_dH0 * 1000 + T * species_dS0
    GHS_residuals.append(GHS_residual)
    
# GHS_residuals

# +
df_nist_tables["GHS residual"] = GHS_residuals

df_nist_tables.head(20)
# -

fig = px.histogram(df_nist_tables, x="GHS residual")
fig.show()

# +
df_nist_tables_inconsistent_all = df_nist_tables[abs(df_nist_tables['GHS residual']) > 200]

df_nist_tables_inconsistent_all.State.value_counts()
# -

# Check the consistency only for valid molecules (that can be formed or composed):

# +
df_nist_tables_inconsistent = df_nist_tables[
    (abs(df_nist_tables['GHS residual']) > 200) & (df_nist_tables['Charge'] == 0.0)
]

df_nist_tables_inconsistent
# -

df_nist_tables_inconsistent["State"].value_counts()

fig = px.histogram(df_nist_tables_inconsistent, x="GHS residual")
fig.show()

# Now, let's check the consistent species:

# +
df_nist_tables_consistent = df_nist_tables[
    (abs(df_nist_tables['GHS residual']) < 50) & (df_nist_tables['Charge'] == 0.0)
]

df_nist_tables_consistent.shape
# -

df_nist_tables_consistent["State"].value_counts()

fig = px.histogram(df_nist_tables_consistent, x="GHS residual")
fig.show()
