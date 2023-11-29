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

# # Machine Learning estimation for std properties
#
# Based on the Molar Mass, State of Matter, and Charge, we now create machine learning models to predict the standard free Gibbs energy of formation, enthalpy, entropy, and heat capacities.

# ## Required dependencies

# +
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from pathlib import Path
import os

DATA_PATH = Path(os.environ.get("DATAPATH"))
sklearn.set_config(transform_output="pandas")
# -

# ## Read preprocessed data

# +
df_nist_stdprops = pd.read_csv(DATA_PATH / "NBS_Tables_preprocessed.csv", index_col=0)

df_nist_stdprops
# -

# ## Organizing the data

# Separating features and targets:

# +
features_columns = ["Molar Mass", "Charge"]
target_columns = ["deltaH0", "deltaG0", "S0", "Cp"]

X = df_nist_stdprops[features_columns]
y = df_nist_stdprops[target_columns]
# -

# Splitting the data:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

X_train

y_train

# ## Data scaling

scaler = StandardScaler()  
scaler.fit(X_train)
X_train_rescaled = scaler.transform(X_train)  
X_test_rescaled = scaler.transform(X_test)  

X_train_rescaled

# ## Build the machine learning model (Neural Network, in this case)

# Initialize the Multilayer Percepton with ADAM:

regr = MLPRegressor(	
    solver='adam',
    learning_rate='adaptive',
    hidden_layer_sizes=(10, 20, 20, 10),  # 10:20:20:10 architecture
    random_state=1, 
    max_iter=10000,  # 10k epochs
    tol=1e-5,
    n_iter_no_change=1000,
    early_stopping=True
)

# Run training:

regr.fit(X_train_rescaled, y_train)

# Run predictions:

y_predict = regr.predict(X_test_rescaled)

y_predict

# +
dict_y_predict = {}
for id_target, target in enumerate(list(y_train.columns)):
    dict_y_predict[target] = y_predict[:, id_target]
    
df_y_predict = pd.DataFrame.from_dict(dict_y_predict)

df_y_predict
# -

# Check the score:

regr.score(X_test_rescaled, y_test)

# ## Assess the results

import matplotlib.pyplot as plt

# Check `deltaH0` results:

# +
plt.figure(figsize=(8, 6))

plt.plot(X_test["Molar Mass"], y_test["deltaH0"], 'o', label="Expected")
plt.plot(X_test["Molar Mass"], df_y_predict["deltaH0"], 'X', label="Prediction")

plt.xlabel("Molar Mass")
plt.ylabel("delta H0")

plt.legend()

plt.show()
# -

# Check `deltaG0` results:

# +
plt.figure(figsize=(8, 6))

plt.plot(X_test["Molar Mass"], y_test["deltaG0"], 'o', label="Expected")
plt.plot(X_test["Molar Mass"], df_y_predict["deltaG0"], 'X', label="Prediction")

plt.xlabel("Molar Mass")
plt.ylabel("delta G0")

plt.legend()

plt.show()
# -

# Check `S0` results:

# +
plt.figure(figsize=(8, 6))

plt.plot(X_test["Molar Mass"], y_test["S0"], 'o', label="Expected")
plt.plot(X_test["Molar Mass"], df_y_predict["S0"], 'X', label="Prediction")

plt.xlabel("Molar Mass")
plt.ylabel("S0")

plt.legend()

plt.show()
# -

# Check `Cp` results:

# +
plt.figure(figsize=(8, 6))

plt.plot(X_test["Molar Mass"], y_test["Cp"], 'o', label="Expected")
plt.plot(X_test["Molar Mass"], df_y_predict["Cp"], 'X', label="Prediction")

plt.xlabel("Molar Mass")
plt.ylabel("Cp")

plt.legend()

plt.show()
