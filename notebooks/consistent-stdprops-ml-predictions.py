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
import multiprocessing as mp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm.auto import tqdm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import skorch
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EpochScoring

from pathlib import Path
import os

DATA_PATH = Path(os.environ.get("DATAPATH"))
sklearn.set_config(transform_output="pandas")

pio.renderers.default = "png"
# -

# Check if PyTorch can use CUDA:

# +
device = "cuda" if torch.cuda.is_available() else "cpu"

device
# -

# ## Read preprocessed data

# ### Loading

# +
df_nist_stdprops = pd.read_csv(DATA_PATH / "NBS_Tables_preprocessed.csv", index_col=0)

df_nist_stdprops
# -

# ### Extending with Se (entropy of the elements of the chemical species) and Num of Elements

# Reading elements data:

# +
df_nist_elements = pd.read_csv(DATA_PATH / "nist_elements.csv", index_col=0)

df_nist_elements


# -

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


# +
Se_species = []
n_elements_in_species = []
for index, row in df_nist_stdprops.iterrows():
    species_formula = row["Formula"]
    elements_in_species = parse_chemical_formula(species_formula)
    
    elements_S0 = 0.0
    n_elements = 0.0
    try:
        for element, count in elements_in_species.items():
            df_element = df_nist_elements.loc[df_nist_elements['element'] == element]
            elements_S0 += df_element['S0'].values[0] * count
            n_elements += count

    except IndexError:
        print(f"Skipping species {species_formula}: element {element} is lacking")
        elements_S0 = np.nan
        n_elements = np.nan
    
    Se_species.append(elements_S0)
    n_elements_in_species.append(n_elements)
    
df_nist_stdprops["Se"] = Se_species
df_nist_stdprops["Num Elements"] = n_elements_in_species
# -

df_nist_stdprops.head(20)

# +
df_nist_stdprops.dropna(inplace=True)

df_nist_stdprops
# -

# ## Organizing the data

# Separating features and targets:

# +
features_columns = ["Molar Mass", "State", "Charge", "Se", "Num Elements"]
target_columns = ["deltaH0", "deltaG0", "S0", "Cp"]

X = df_nist_stdprops[features_columns]
y = df_nist_stdprops[target_columns]
# -

# ### Encoding the State of Matter feature

# Let's put together state of matter with few occurences (unsure if this is a good approach):

X["State"].value_counts()

state_renamings = {
    "g2": "g",
    "cr3": "cr",
    "l2": "l",
    "g3": "g",
    "cr4": "cr",
    "l3": "l",
}
# X.replace(state_renamings, inplace=True)

# +
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X[["State"]])

encoder_categories = list(encoder.categories_[0])

encoder_categories

# +
X_state_encoded = encoder.transform(X[["State"]])

X_state_encoded

# +
X_encoded = pd.concat([X, X_state_encoded], axis=1)
X_encoded.drop(columns=["State"], inplace=True)

X_encoded
# -

# ### Splitting the data

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, test_size=test_size)

X_train

y_train

# ## Data scaling

scaler = StandardScaler()  
scaler.fit(X_train)
X_train_rescaled = scaler.transform(X_train)  
X_test_rescaled = scaler.transform(X_test)  

X_train_rescaled

# ## Build the machine learning model (Neural Network, in this case)

# ### MLP model setup with `skorch`

# +
num_features = int(X_train_rescaled.shape[1])


class NetArchitecture(nn.Module):
    """
    The neural net achitecture setup.
    """
    def __init__(self):
        super(NetArchitecture, self).__init__()
        self.input_layer = nn.Linear(num_features, 20)
        self.hidden_layer1 = nn.Linear(20, 30)
        self.hidden_layer2 = nn.Linear(30, 20)
        self.hidden_layer3 = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, 4)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x


class CustomNetThermodynamicInformed(NeuralNetRegressor):
    def __init__(self, *args, lambda1=1e-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = nn.MSELoss()(y_pred, y_true)
        
        dG0 = torch.mul(y_pred[:, 1], 1000)
        dH0 = torch.mul(y_pred[:, 0], 1000)
        T = 298.15
        dS0 = torch.mul(y_pred[:, 2], T)
        
        # Undo scaling applied to Se values to properly compute the residuals
        X_as_numpy = scaler.inverse_transform(X.numpy())
        Se_torch_tensor = torch.from_numpy(X_as_numpy[:, 2])
        dSe = torch.mul(Se_torch_tensor, T)
        
        # Compute the GHS residual
        GHS_residual = dG0 - dH0 + dS0 - dSe
        custom_term = torch.norm(GHS_residual, p=2) / y_pred.size(0)
        
        loss += self.lambda1 * custom_term
        return loss



# -

# Convenient callback functions:

# * R2 score to compare with `sklearn` MLP:

r2_scoring = EpochScoring(
    scoring='r2', 
    lower_is_better=False,
    on_train=False,
    name='valid_r2'
)


# * Progress bar callback:

class TqdmCallback(Callback):
    def on_train_begin(self, net, X, y, **kwargs):
        self.pbar = tqdm(total=net.max_epochs)

    def on_epoch_end(self, net, **kwargs):
        self.pbar.update(1)
        epoch = net.history[-1]
        self.pbar.set_postfix({
            'loss': epoch['train_loss'],
            'valid_r2': epoch['valid_r2'] if 'valid_r2' in epoch else 'N/A'
        })

    def on_train_end(self, net, X, y, **kwargs):
        self.pbar.close()


# * Early stopping callback:

max_epochs = 20000
# max_epochs = 10
rel_error_stop_criterion = 1e-8
min_percentage_of_num_epochs = 0.2
early_stopping = skorch.callbacks.EarlyStopping(
    patience=int(min_percentage_of_num_epochs * max_epochs), 
    threshold=rel_error_stop_criterion
)

# ### Hyper-parameter optimization
#
# Before run a full NN models, let's find the best parameters to configure our NN model beforehand.

# * The reduced NN model:

# +
max_epochs_gs = 3000
rel_error_stop_criterion_gs = 1e-5
min_percentage_of_num_epochs_gs = 0.1
early_stopping_gs = skorch.callbacks.EarlyStopping(
    patience=int(min_percentage_of_num_epochs_gs * max_epochs_gs), 
    threshold=rel_error_stop_criterion_gs
)


net_gs_fit = CustomNetThermodynamicInformed(
    module=NetArchitecture,
    lambda1=1e1,
    max_epochs=max_epochs_gs,
    lr=1e-2,
    batch_size=X_train_rescaled.shape[0],
    optimizer=torch.optim.Adam,
    callbacks=[early_stopping_gs],
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=False
)
# -

# * Setting the Randomized Search Cross-Validation to explore the parameters in a 3-folds setting:

ss_generator = ShuffleSplit(n_splits=3, test_size=test_size, random_state=1)

# +
lr_values = np.random.uniform(1e-5, 1e-1, 25).tolist()
lambda1_values = np.random.uniform(0.001, 5e1, 25).tolist()
params = {
    'lr': lr_values,
    'lambda1': lambda1_values,
}

gs = RandomizedSearchCV(net_gs_fit, params, cv=ss_generator, n_iter=10, random_state=42, verbose=3)
# -

# * Search the best parameters using the reduced model:

# +
X_full_rescaled = scaler.transform(X_encoded)

gs.fit(
    X_full_rescaled.to_numpy().astype(np.float32),
    y.to_numpy().astype(np.float32)
)

# +
df_parameter_search = pd.DataFrame.from_dict(gs.cv_results_)

df_parameter_search
# -

# * Collecting the results:

# +
best_lambda1 = gs.best_params_['lambda1']
best_lr = gs.best_params_['lr']

print(f"Best lambda1 = {best_lambda1}\t Best lr = {best_lr}")
# -

# ### Setting the complete NN model:

# Initialize the NN with skorch:

torch.manual_seed = 42
net = CustomNetThermodynamicInformed(
    module=NetArchitecture,
    lambda1=best_lambda1,
    max_epochs=max_epochs,
    lr=best_lr,
    batch_size=X_train_rescaled.shape[0],
    optimizer=torch.optim.Adam,
    callbacks=[r2_scoring, TqdmCallback(), early_stopping],
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=False
)

X_torch = torch.from_numpy(X_train_rescaled.to_numpy()).float()
y_torch = torch.from_numpy(y_train.to_numpy()).float()

# ### Training/testing the model

# Perform training:

net.fit(X_torch, y_torch)

# Run predictions:

X_test_torch = torch.from_numpy(X_test_rescaled.to_numpy()).float()
y_predict = net.predict(X_test_torch)

# +
dict_y_predict = {}
for id_target, target in enumerate(list(y_train.columns)):
    dict_y_predict[target] = y_predict[:, id_target]
    
df_y_predict = pd.DataFrame.from_dict(dict_y_predict)

df_y_predict
# -

# Check the score:

net.score(
    X_test_rescaled.to_numpy().astype(np.float32), 
    y_test.to_numpy().astype(np.float32)
)

# ### Evaluate the loss function evolution through the epochs

# +
history = net.history
loss_history = history[:, 'valid_loss']
r2_history = history[:, 'valid_r2']

df_loss_history = pd.DataFrame.from_dict(
    {
        "Epoch": list(range(1, len(loss_history) + 1)), 
        "Loss function evaluations": loss_history,
        "R2 score": r2_history
    }
)

df_loss_history

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_loss_history["Epoch"], 
    y=df_loss_history["Loss function evaluations"],
    mode='lines',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Evolution of the Loss Function through Epochs",
    xaxis_title="Epoch",
    yaxis_title="Loss function evaluation",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="right",
    )
)

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_loss_history["Epoch"], 
    y=df_loss_history["R2 score"],
    mode='lines',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Evolution of the R2-score through Epochs (on validation)",
    xaxis_title="Epoch",
    yaxis_title="R2 score",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="right",
    )
)
fig.update_yaxes(range=[0.0, 1.0])

fig.show()
# -

# ### MLP model setup with `sklearn`

regr = MLPRegressor(	
    solver='adam',
    learning_rate='adaptive',
    hidden_layer_sizes=(20, 30, 20, 10),  # 20:30:20:10 architecture
    random_state=1, 
    max_iter=max_epochs,
    tol=rel_error_stop_criterion,
    n_iter_no_change=int(min_percentage_of_num_epochs * max_epochs),
    early_stopping=True
)

regr.fit(X_train_rescaled, y_train)

# +
y_predict_sklearn = regr.predict(X_test_rescaled)

dict_y_predict_sklearn = {}
for id_target, target in enumerate(list(y_train.columns)):
    dict_y_predict_sklearn[target] = y_predict_sklearn[:, id_target]
    
df_y_predict_sklearn = pd.DataFrame.from_dict(dict_y_predict_sklearn)

df_y_predict_sklearn
# -

regr.score(X_test_rescaled, y_test)

# +
loss_evaluations_sklearn = regr.loss_curve_
df_loss_history_sklearn = pd.DataFrame.from_dict(
    {
        "Epoch": list(range(1, len(loss_evaluations_sklearn) + 1)), 
        "Loss function evaluations": loss_evaluations_sklearn
    }
)

df_loss_history_sklearn

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_loss_history_sklearn["Epoch"], 
    y=df_loss_history_sklearn["Loss function evaluations"],
    mode='lines',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Evolution of the Loss Function through Epochs (sklearn)",
    xaxis_title="Epoch",
    yaxis_title="Loss function evaluation",
    font=dict(
        size=18,
    ),
)

fig.show()
# -

# ## Assess the results

# ### Collecting results

# +
target_errors = {}

for target_name in list(y_test.columns):
    target_abs_error = np.abs(y_test[target_name].values - df_y_predict[target_name].values)
    target_errors[f"{target_name} abs. error"] = target_abs_error
    
    target_rel_error = target_abs_error / np.abs(df_y_predict[target_name].values)
    target_errors[f"{target_name} rel. error"] = target_rel_error
    
df_target_errors = pd.DataFrame.from_dict(target_errors)
df_target_errors

# +
target_results = {}

for target_name in list(y_test.columns):
    target_results[f"{target_name} predicted"] = df_y_predict[target_name].values
    target_results[f"{target_name} expected"] = y_test[target_name].values
    
df_target_results= pd.DataFrame.from_dict(target_results)
df_target_results
# -

# ### Check `deltaH0` results

# Against Molar Mass:

# +
fig = go.Figure()

fig_expected = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaH0 expected"],
    mode='markers',
    name='Expected'
)

fig_predicted = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaH0 predicted"],
    mode='markers',
    name='Predicted'
)
fig.add_traces([fig_expected, fig_predicted])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="deltaH0",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="right",
    )
)

fig.show()
# -

# Against Expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaH0 expected"], 
    y=df_target_results["deltaH0 predicted"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaH0 expected"], 
    y=df_target_results["deltaH0 expected"],
    mode='lines',
    line=dict(color="black", dash='dash'),
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted values for deltaH0",
    xaxis_title="Actual deltaH0 values",
    yaxis_title="Predicted deltaH0 values",
    showlegend=False,
    font=dict(
        size=18,
    )
)

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaH0 predicted"], 
    y=df_target_errors["deltaH0 rel. error"],
    mode='markers',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Rel. error vs Predicted values for delta H0",
    xaxis_title="Predicted delta H0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) H0 values",
    showlegend=False,
    font=dict(
        size=14,
    )
)

fig.show()
# -

# `deltaH0` residuals distribution:

fig = px.histogram(df_target_errors, x="deltaH0 rel. error", nbins=80)
fig.show()

# ### Checking `deltaG0` results

# Against molar mass:

# +
fig = go.Figure()

fig_expected = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaG0 expected"],
    mode='markers',
    name='Expected'
)

fig_predicted = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaG0 predicted"],
    mode='markers',
    name='Predicted'
)
fig.add_traces([fig_expected, fig_predicted])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="deltaG0",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="right",
    )
)

fig.show()
# -

# Against Expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaG0 expected"], 
    y=df_target_results["deltaG0 predicted"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaG0 expected"], 
    y=df_target_results["deltaG0 expected"],
    line=dict(color="black", dash='dash'),
    mode='lines',
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted values for deltaG0",
    xaxis_title="Actual deltaG0 values",
    yaxis_title="Predicted deltaG0 values",
    showlegend=False,
    font=dict(
        size=18,
    )
)

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaG0 predicted"], 
    y=df_target_errors["deltaG0 rel. error"],
    mode='markers',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Rel. error vs Predicted values for delta G0",
    xaxis_title="Predicted delta G0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) G0 values",
    showlegend=False,
    font=dict(
        size=14,
    )
)

fig.show()
# -

# `deltaG0` residuals distribution:

fig = px.histogram(df_target_errors, x="deltaG0 rel. error", nbins=80)
fig.show()

# ### Checking `S0` results

# Checking against Molar Mass:

# +
fig = go.Figure()

fig_expected = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["S0 expected"],
    mode='markers',
    name='Expected'
)

fig_predicted = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["S0 predicted"],
    mode='markers',
    name='Predicted'
)
fig.add_traces([fig_expected, fig_predicted])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="S0",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="left",
        x=0.025,
        y=0.95,
    )
)

fig.show()
# -

# Checking against expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["S0 expected"], 
    y=df_target_results["S0 predicted"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["S0 expected"], 
    y=df_target_results["S0 expected"],
    line=dict(color="black", dash='dash'),
    mode='lines',
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted values for S0",
    xaxis_title="Actual S0 values",
    yaxis_title="Predicted S0 values",
    showlegend=False,
    font=dict(
        size=18,
    )
)

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["S0 predicted"], 
    y=df_target_errors["S0 rel. error"],
    mode='markers',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Rel. error vs Predicted values for S0",
    xaxis_title="Predicted S0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) S0 values",
    showlegend=False,
    font=dict(
        size=14,
    )
)

fig.show()
# -

# `S0` residuals distribution:

fig = px.histogram(df_target_errors, x="S0 rel. error")
fig.show()

# ### Check `Cp` results

# Checking against molar mass:

# +
fig = go.Figure()

fig_expected = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["Cp expected"],
    mode='markers',
    name='Expected'
)

fig_predicted = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["Cp predicted"],
    mode='markers',
    name='Predicted'
)
fig.add_traces([fig_expected, fig_predicted])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="Cp",
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="left",
        x=0.025,
        y=0.95,
    )
)

fig.show()
# -

# Checking against expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["Cp expected"], 
    y=df_target_results["Cp predicted"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["Cp expected"], 
    y=df_target_results["Cp expected"],
    line=dict(color="black", dash='dash'),
    mode='lines',
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted values for Cp",
    xaxis_title="Actual Cp values",
    yaxis_title="Predicted Cp values",
    showlegend=False,
    font=dict(
        size=18,
    ),
)

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["Cp predicted"], 
    y=df_target_errors["Cp rel. error"],
    mode='markers',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Rel. error vs Predicted values for Cp",
    xaxis_title="Predicted Cp values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) Cp values",
    showlegend=False,
    font=dict(
        size=14,
    )
)

fig.show()
# -

# `Cp` residuals distribution:

fig = px.histogram(df_target_errors, x="Cp rel. error", nbins=80)
fig.show()

# ### Verifying the consistency of the predicted values
#
# To evaluate the quality of the predictions from a thermodynamic point of view, we should assess the GHS residual to check if the predictions are consistent.

# * `skorch`

# +
df_predicted_species = df_nist_stdprops.loc[X_test_rescaled.index, ["Formula", "Molar Mass", "Se"]]
for target in list(df_y_predict.columns):
    df_predicted_species.loc[:, target] = df_y_predict.loc[:, target].values
    
df_predicted_species
# -

# * `sklearn`:

# +
df_predicted_species_sklearn = df_nist_stdprops.loc[X_test_rescaled.index, ["Formula", "Molar Mass", "Se"]]
for target in list(df_y_predict_sklearn.columns):
    df_predicted_species_sklearn.loc[:, target] = df_y_predict_sklearn.loc[:, target].values
    
df_predicted_species_sklearn
# -

# Collecting the GHS residuals:

# +
T = 298.15  # in K
predicted_GHS_residuals = []
predicted_GHS_residuals_sklearn = []
expected_GHS_residuals = []
df_expected_stdprops = df_nist_stdprops.loc[X_test_rescaled.index, :]
for index, row in df_predicted_species.iterrows():
    # Skorch
    G0_predicted = row["deltaG0"] * 1000
    H0_predicted = row["deltaH0"] * 1000
    S0_predicted = row["S0"]
    Se_predicted = row["Se"]
    GHS_residual_predicted = G0_predicted - H0_predicted + T * (S0_predicted - Se_predicted)
    predicted_GHS_residuals.append(GHS_residual_predicted)
    
    # Sklearn
    G0_predicted_sklearn = df_predicted_species_sklearn.loc[index, "deltaG0"] * 1000
    H0_predicted_sklearn = df_predicted_species_sklearn.loc[index, "deltaH0"] * 1000
    S0_predicted_sklearn = df_predicted_species_sklearn.loc[index, "S0"]
    Se_predicted_sklearn = df_predicted_species_sklearn.loc[index, "Se"]
    GHS_residual_predicted_sklearn = G0_predicted_sklearn - H0_predicted_sklearn + T * (S0_predicted_sklearn - Se_predicted_sklearn)
    predicted_GHS_residuals_sklearn.append(GHS_residual_predicted_sklearn)
    
    G0_expected = df_expected_stdprops.loc[index, "deltaG0"] * 1000
    H0_expected = df_expected_stdprops.loc[index, "deltaH0"] * 1000
    S0_expected = df_expected_stdprops.loc[index, "S0"]
    Se_expected = df_expected_stdprops.loc[index, "Se"]
    GHS_residual_expected = G0_expected - H0_expected + T * (S0_expected - Se_expected)
    expected_GHS_residuals.append(GHS_residual_expected)
    
df_predicted_species["GHS residual"] = predicted_GHS_residuals
df_predicted_species_sklearn["GHS residual"] = predicted_GHS_residuals_sklearn
df_expected_stdprops["GHS residual"] = expected_GHS_residuals
df_predicted_species
# -

# #### Visual verification

# * `skorch`

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_expected_stdprops["GHS residual"], 
    y=df_predicted_species["GHS residual"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_expected_stdprops["GHS residual"], 
    y=df_expected_stdprops["GHS residual"], 
    line=dict(color="black", dash='dash'),
    mode='lines',
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted GHS residuals",
    xaxis_title="Actual GHS residuals",
    yaxis_title="Predicted GHS residuals",
    showlegend=False,
    font=dict(
        size=18,
    )
)

fig.show()
# -

# * `sklearn`:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_expected_stdprops["GHS residual"], 
    y=df_predicted_species_sklearn["GHS residual"],
    mode='markers',
)

fig2 = go.Scatter(
    x=df_expected_stdprops["GHS residual"], 
    y=df_expected_stdprops["GHS residual"], 
    line=dict(color="black", dash='dash'),
    mode='lines',
)
fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Actual vs Predicted GHS residuals",
    xaxis_title="Actual GHS residuals",
    yaxis_title="Predicted GHS residuals",
    showlegend=False,
    font=dict(
        size=18,
    )
)

fig.show()
# -

# * `skorch` (consistent) vs `sklearn` (inconsistent):

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_predicted_species_sklearn["GHS residual"], name='Unconstrained')
fig2 = go.Histogram(x=df_predicted_species["GHS residual"], name='Constrained')

fig.add_traces([fig1, fig2])

fig.update_layout(
    # title="GHS residuals",
    xaxis_title="GHS residuals",
    barmode='overlay',
    # yaxis_title="Predicted GHS residuals",
    # showlegend=False,
    font=dict(
        size=18,
    ),
    legend=dict(
        yanchor="top",
        xanchor="left",
        x=0.025,
        y=0.95,
    )
)
fig.update_traces(opacity=0.75)
fig.add_vline(x=0.0, line_width=3, line_dash="dash", line_color="black")

fig.show()
