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
from pathlib import Path
import os

DATA_PATH = Path(os.environ.get("DATAPATH"))
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
# df_nist_tables_full.replace("-", np.nan, inplace=True)
# df_nist_tables_full.replace(r"\*", "", inplace=True)  # remove asterisk highlights as well
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

# cr     711
#
# g      531
#
# ai      99
#
# l       75
#
# ao      38
#
# cr2     27
#
# g2      14

import plotly.express as px

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


