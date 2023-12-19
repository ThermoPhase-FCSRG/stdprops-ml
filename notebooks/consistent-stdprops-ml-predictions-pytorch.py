# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: stdprops
#     language: python
#     name: python3
# ---

# # Machine Learning estimation for std properties
#
# Based on the Molar Mass, State of Matter, Charge, number of Elements, and entropy of elements, we now create machine learning models to predict the standard free Gibbs energy of formation, enthalpy, entropy, and heat capacities.

# ## Required dependencies

# +
import os
# Should be set before importing numpy
# This setup decreased the run times by near 3 times
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import multiprocessing as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import random
from tqdm.auto import tqdm

import sklearn
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import skorch
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, EpochScoring

from pathlib import Path

DATA_PATH = Path(os.environ.get("DATAPATH"))
NOTEBOOKS_PATH = Path(os.environ.get("NOTEBOOKSPATH"))
sklearn.set_config(transform_output="pandas")

pio.renderers.default = "png"

# For reproducibility's sake
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# -

# Create Paths to store the outputs:

RESULTS_PATH = Path(NOTEBOOKS_PATH / "outputs" / "consistent_ml_torch")
RESULTS_PATH.mkdir(
    parents=True, exist_ok=True,
)
is_saving_figure_enabled = True

# Set the number of threads to be used by `PyTorch`:

available_cpus = mp.cpu_count()
parallel_jobs = available_cpus - 2 if available_cpus > 2 else 1
torch.set_num_threads(parallel_jobs)

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
X.replace(state_renamings, inplace=True)

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
num_targets = int(y_train.shape[1])


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
        self.output_layer = nn.Linear(10, num_targets)
        
        # Use the same initial weights and biases as tensorflow
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        '''
        Convenient automatic initialization of weights and biases.
        
        This sets the same initialization used in TensorFlow.
        Good to compare the performance.
        '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with Xavier uniform (Glorot uniform)
                nn.init.xavier_uniform_(module.weight)
                # Initialize bias to zeros
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x


class CustomNetThermodynamicInformed(NeuralNetRegressor):
    def __init__(
        self, *args, lambda1=1e-1, negative_loss=False, use_r2_score=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1
        self.negative_loss = negative_loss
        self.use_r2_score = use_r2_score

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = self.criterion_(y_pred, y_true)
        
        # Extract the actual data from the Skorch Dataset
        if isinstance(X, skorch.dataset.Dataset):
            X = X.X  # Access the underlying numpy array
        
        dG0 = torch.mul(y_pred[:, 1], 1000).float()
        dH0 = torch.mul(y_pred[:, 0], 1000).float()
        T = 298.15
        dS0 = torch.mul(y_pred[:, 2], T).float()
        
        # Undo scaling applied to Se values to properly compute the residuals
        X_as_numpy = scaler.inverse_transform(X)
        Se_torch_tensor = torch.from_numpy(X_as_numpy[:, 2]).float()
        dSe = torch.mul(Se_torch_tensor, T)
        
        # Compute the GHS residual
        GHS_residual_left_hand_side = dG0
        GHS_residual_right_hand_side = dH0 - (dS0 - dSe)
        GHS_residual = GHS_residual_left_hand_side - GHS_residual_right_hand_side
        custom_term = torch.norm(GHS_residual, p=2) / y_pred.size(0)
        
        loss += self.lambda1 * custom_term
        return loss
    
    def score(self, X, y):
        loss = np.nan
        self.check_is_fitted()
        X = self.get_dataset(X)
        y = torch.as_tensor(y, device=self.device)
        y_pred = self.predict(X)
        
        if self.use_r2_score:
            loss = r2_score(y, y_pred)
        else:
            y_pred = torch.as_tensor(y_pred, device=self.device)

            # Compute the custom loss
            loss = self.get_loss(y_pred, y, X=X, training=False)
            loss = -loss.item() if self.negative_loss else loss.item()
            
        return loss



# -

# Plotting the NN architecture:

# +
import matplotlib.pyplot as plt


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, input_labels, output_labels):
    '''
    Draw a neural network cartoon using matplotlib.
    
    :param ax: matplotlib.axes.Axes instance
    :param left: float, leftmost node center x-coordinate
    :param right: float, rightmost node center x-coordinate
    :param bottom: float, bottommost node center y-coordinate
    :param top: float, topmost node center y-coordinate
    :param layer_sizes: list of int, size of each layer
    :param input_labels: list of str, labels for input features
    :param output_labels: list of str, labels for output targets
    '''
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle(
                (n*h_spacing + left, layer_top - m*v_spacing), 
                v_spacing/4.,
                color='w', 
                ec='k', 
                zorder=4
            )
            ax.add_artist(circle)
            # Annotation for the input layer
            if n == 0 and m < len(input_labels):
                plt.text(left - 0.025, layer_top - m*v_spacing - 0.005, input_labels[m], fontsize=12, ha='right')
            # Annotation for hidden layers
            elif n < len(layer_sizes) - 1:
                annotation_y_position = layer_top + 0.035
                plt.text(n*h_spacing + left, annotation_y_position, f'Hidden Layer {n}\n({layer_size} neurons)\nReLU', fontsize=12, ha='center')
            # Annotation for the output layer
            elif n == len(layer_sizes) - 1 and m < len(output_labels):
                plt.text(n*h_spacing + left + 0.025, layer_top - m*v_spacing - 0.005, output_labels[m], fontsize=12, ha='left')
    
    # Connect
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

    # GHS Residual Equation Annotation
    ghs_residual_text = r"$\Delta_{\mathrm{f}} G^{\circ}=\Delta_{\mathrm{f}} H^{\circ}-T \Delta_{\mathrm{f}} S^{\circ}$"
    plt.text(right + 0.15, top - 0.3, ghs_residual_text, fontsize=14, ha='right')
    plt.text(right + 0.1, top - 0.27, "Satisfying:", fontsize=14, ha='right')
    
    # LaTeX Annotations for the objective function
    error_function_text = r"$f_{\text{error}} = \dfrac{1}{N_{exp}}\sum_i^{N_{exp}}" \
                          r"\left[(\Delta G^{\circ}_{i, sim} -  \Delta G^{\circ}_{i, exp})^2" \
                          r"+ (\Delta H^{\circ}_{i, sim} -  \Delta H^{\circ}_{i, exp})^2" \
                          r"+ (S^{\circ}_{i, sim} -  S^{\circ}_{i, exp})^2" \
                          r"+ (Cp^{\circ}_{i, sim} -  Cp^{\circ}_{i, exp})^2\right]$"
    thermo_function_text = r"$f_{\text{thermo}} = \dfrac{\lambda}{N_{exp}}\sum_i^{N_{exp}}" \
                           r"(\Delta G^{\circ}_{i, sim} - \Delta H^{\circ}_{i, sim} + T \Delta S^{\circ}_{i, sim})^2$"
    ax.text(0.5, 0.1, error_function_text, fontsize=14, ha='center', va='top', transform=ax.transAxes, color='red')
    ax.text(0.5, 0.055, thermo_function_text, fontsize=14, ha='center', va='top', transform=ax.transAxes, color='blue')
    
    # Objective Function Annotation with Different Colors
    obj_function_text_1 = r"$f_{\text{obj}} = $"
    obj_function_text_2 = r"$f_{\text{error}}$"
    obj_function_text_3 = r"$ + $"
    obj_function_text_4 = r"$f_{\text{thermo}}$"

    ax.text(0.5, -0.01, obj_function_text_1, fontsize=14, ha='right', va='top', transform=ax.transAxes)
    ax.text(0.505, -0.01, obj_function_text_2, fontsize=14, ha='left', va='top', transform=ax.transAxes, color='red')
    ax.text(0.542, -0.01, obj_function_text_3, fontsize=14, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.56, -0.01, obj_function_text_4, fontsize=14, ha='left', va='top', transform=ax.transAxes, color='blue')
    # obj_function_text = r"$f_{\text{obj}} = f_{\text{error}} + f_{\text{thermo}}$"
    # ax.text(0.5, 0.0, obj_function_text, fontsize=14, ha='center', va='top', transform=ax.transAxes)


# Define the layer sizes and labels
layer_sizes = [11, 20, 30, 20, 10, 4]  # Adjust as needed
input_labels = ['Molar Mass', 'Charge', 'Se', 'Num Elements', 'State_ai', 'State_am', 'State_ao', 'State_cr', 'State_cr2', 'State_g', 'State_l']
output_labels = [r'$\Delta H^0$', r'$\Delta G^0$', r'$S^0$', r'$C_p$']

# Create the figure
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, input_labels, output_labels)

plt.tight_layout()
if is_saving_figure_enabled:
    plt.savefig(RESULTS_PATH / "nn_architecture.png", dpi=600)

plt.show()
# -

# The design of the objective-function is described below. The term $f_{\text{thermo}}$ is responsible to integrate the thermodynamic knowledge into the Neural Network fitting procedure.
#
# \begin{equation}
# f_{\text{error}} = 
# \dfrac{1}{N_{exp}}\sum_i^{N_{exp}}
# \left[
#     (\Delta G^{\circ}_{i, sim} -  \Delta G^{\circ}_{i, exp})^2
#     + (\Delta H^{\circ}_{i, sim} -  \Delta H^{\circ}_{i, exp})^2
#     + (S^{\circ}_{i, sim} -  S^{\circ}_{i, exp})^2
#     + (Cp^{\circ}_{i, sim} -  Cp^{\circ}_{i, exp})^2
# \right]
# \end{equation}
#
# \begin{equation}
# f_{\text{thermo}} = 
# \dfrac{\lambda}{N_{exp}}\sum_i^{N_{exp}}
#     (\Delta G^{\circ}_{i, sim} - \Delta H^{\circ}_{i, sim} + T \Delta S^{\circ}_{i, sim})^2
# \end{equation}
#
# \begin{equation}
# f_{\text{obj}} = f_{\text{error}} + f_{\text{thermo}} 
# \end{equation}
#
# In the $f_{\text{thermo}}$ contribution, $\lambda$ is a penalty parameter enforcing the thermodynamic relation to be satisfied. In this work, we adopt $\lambda = 10$ through experimentation.

# Convenient callback functions:

# * R2 score callback:

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
rel_error_stop_criterion = 1e-8
min_percentage_of_num_epochs = 0.1
early_stopping = skorch.callbacks.EarlyStopping(
    patience=int(min_percentage_of_num_epochs * max_epochs), 
    threshold=rel_error_stop_criterion
)


# ### Hyper-parameter optimization
#
# Before run a full NN models, let's find the best parameters to configure our NN model beforehand.

# * Convenient custom score considering the implemented custom loss function in the MLP:

def custom_loss_scorer(net, X, y):
    # Predict using the provided net
    y_pred = net.predict(X)

    # Convert y and y_pred to PyTorch tensors
    y_true_tensor = torch.as_tensor(y, dtype=torch.float32)
    y_pred_tensor = torch.as_tensor(y_pred, dtype=torch.float32)

    # Compute the custom loss using the net's get_loss method
    # Assuming X is already a numpy array and not a Skorch Dataset here
    loss = net.get_loss(y_pred_tensor, y_true_tensor, X=X, training=False)

    # Return the negative loss since lower is better
    return -loss.item()


# * The reduced NN model:

# +
max_epochs_gs = 3000
rel_error_stop_criterion_gs = 1e-6
min_percentage_of_num_epochs_gs = 0.1
early_stopping_gs = skorch.callbacks.EarlyStopping(
    patience=int(min_percentage_of_num_epochs_gs * max_epochs_gs), 
    threshold=rel_error_stop_criterion_gs
)

lambda1 = 1e1
model_gs = NetArchitecture()
net_gs_fit = CustomNetThermodynamicInformed(
    module=model_gs,
    criterion=nn.MSELoss(),
    lambda1=lambda1,
    # The sklearn hyperparam optimimization is a maximization of the score.
    # Thus, since the loss is a MSE, we need to use loss = -loss.
    # This is only need in the hyperparam optimization, pytorch works
    # with the minimization of the loss function.
    negative_loss=True,
    use_r2_score=True,
    max_epochs=max_epochs_gs,
    lr=1e-2,
    batch_size=X_train_rescaled.shape[0],
    optimizer=torch.optim.Adam,
    callbacks=[early_stopping_gs],
    device='cpu',
    verbose=False
)
# -

# * Setting the Randomized Search Cross-Validation to explore the parameters in a 4-folds setting:

ss_generator = ShuffleSplit(n_splits=4, test_size=test_size, random_state=1)

# +
lr_values = np.random.uniform(1e-5, 2e-1, 30).tolist()
# lambda1_values = np.random.uniform(0.1, 1e2, 30).tolist()  # not a hyper-parameter!
params = {
    'lr': lr_values,
    # 'lambda1': lambda1_values,  # in fact, this is not a hyper-parameter since it changes the fobj
}

gs = RandomizedSearchCV(
    net_gs_fit, 
    params, 
    scoring={"r2": make_scorer(r2_score), "Constrained Loss": custom_loss_scorer},
    refit="Constrained Loss",
    cv=ss_generator, 
    n_iter=10, 
    random_state=42, 
    verbose=3
)
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
# best_lambda1 = gs.best_params_['lambda1']  # not a hyper-parameter!
best_lambda1 = lambda1
best_lr = gs.best_params_['lr']

print(f"Best lambda1 = {best_lambda1}\t Best lr = {best_lr}")
# -

# ### Setting the complete NN model:

# Initialize the NN with skorch:

net = CustomNetThermodynamicInformed(
    module=NetArchitecture,
    criterion=nn.MSELoss(),
    lambda1=best_lambda1,
    max_epochs=max_epochs,
    lr=best_lr,
    batch_size=X_train_rescaled.shape[0],
    optimizer=torch.optim.Adam,
    callbacks=[r2_scoring, TqdmCallback(), early_stopping],
    # device='cuda' if torch.cuda.is_available() else 'cpu',
    device='cpu',
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

r2_score(y_test, y_predict)

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

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "loss_constrained.png")
    
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

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "r2_constrained.png")
    
fig.show()
# -

# ### MLP model setup with `skorch`/`pytorch` without thermodynamical constrains

net_unconstrained = NeuralNetRegressor(
    module=NetArchitecture,
    criterion=nn.MSELoss(),
    max_epochs=max_epochs,
    lr=best_lr,
    batch_size=X_train_rescaled.shape[0],
    optimizer=torch.optim.Adam,
    callbacks=[r2_scoring, TqdmCallback(), early_stopping],
    # device='cuda' if torch.cuda.is_available() else 'cpu',
    device='cpu',
    verbose=False
)

net_unconstrained.fit(X_torch, y_torch)

# +
y_predict_unconstrained = net_unconstrained.predict(X_test_torch)

dict_y_predict_unconstrained = {}
for id_target, target in enumerate(list(y_train.columns)):
    dict_y_predict_unconstrained[target] = y_predict_unconstrained[:, id_target]
    
df_y_predict_unconstrained = pd.DataFrame.from_dict(dict_y_predict_unconstrained)

df_y_predict_unconstrained
# -

r2_score(y_test, y_predict_unconstrained)

# +
history_unconstrained = net_unconstrained.history
loss_history_unconstrained = history_unconstrained[:, 'valid_loss']
r2_history_unconstrained = history_unconstrained[:, 'valid_r2']

df_loss_history_unconstrained = pd.DataFrame.from_dict(
    {
        "Epoch": list(range(1, len(loss_history_unconstrained) + 1)), 
        "Loss function evaluations": loss_history_unconstrained,
        "R2 score": r2_history_unconstrained
    }
)

df_loss_history_unconstrained

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_loss_history_unconstrained["Epoch"], 
    y=df_loss_history_unconstrained["Loss function evaluations"],
    mode='lines',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Evolution of the Loss Function through Epochs (unconstrained)",
    xaxis_title="Epoch",
    yaxis_title="Loss function evaluation",
    font=dict(
        size=18,
    ),
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "loss_unconstrained.png")

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_loss_history_unconstrained["Epoch"], 
    y=df_loss_history_unconstrained["R2 score"],
    mode='lines',
)

fig.add_traces([fig1])

fig.update_layout(
    title="Evolution of the R2-score through Epochs (unconstrained)",
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

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "r2_unconstrained.png")

fig.show()
# -

# ## Assess the results

# ### Collecting results

# * Thermodynamics-Informed:

# +
target_errors = {}

for target_name in list(y_test.columns):
    target_abs_error = np.abs(y_test[target_name].values - df_y_predict[target_name].values)
    target_errors[f"{target_name} abs. error"] = target_abs_error
    
    target_rel_error = target_abs_error / np.abs(df_y_predict[target_name].values)
    target_errors[f"{target_name} rel. error"] = target_rel_error
    
df_target_errors = pd.DataFrame.from_dict(target_errors)
df_target_errors
# -

# * Unconstrained:

# +
target_errors_unconstrained = {}

for target_name in list(y_test.columns):
    target_abs_error = np.abs(y_test[target_name].values - df_y_predict_unconstrained[target_name].values)
    target_errors_unconstrained[f"{target_name} abs. error"] = target_abs_error
    
    target_rel_error = target_abs_error / np.abs(df_y_predict_unconstrained[target_name].values)
    target_errors_unconstrained[f"{target_name} rel. error"] = target_rel_error
    
df_target_errors_unconstrained = pd.DataFrame.from_dict(target_errors_unconstrained)
df_target_errors_unconstrained
# -

# * Assembling the predictions:

# +
target_results = {}

for target_name in list(y_test.columns):
    target_results[f"{target_name} predicted"] = df_y_predict[target_name].values
    target_results[f"{target_name} predicted (unconstrained)"] = df_y_predict_unconstrained[target_name].values
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

fig_unconstrained = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaH0 predicted (unconstrained)"],
    mode='markers',
    name='Predicted (unconstrained)'
)

fig.add_traces([fig_unconstrained, fig_predicted, fig_expected])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="deltaH0",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "deltaH0_scatter.png")

fig.show()
# -

# Against Expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaH0 expected"], 
    y=df_target_results["deltaH0 predicted"],
    name='GHS constrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaH0 expected"], 
    y=df_target_results["deltaH0 predicted (unconstrained)"],
    name='Unconstrained',
    mode='markers',
)

fig3 = go.Scatter(
    x=df_target_results["deltaH0 expected"], 
    y=df_target_results["deltaH0 expected"],
    name='Actual',
    mode='lines',
    line=dict(color="black", dash='dash'),
)

fig.add_traces([fig2, fig1, fig3])

fig.update_layout(
    title="Actual vs Predicted values for deltaH0",
    xaxis_title="Actual deltaH0 values",
    yaxis_title="Predicted deltaH0 values",
    showlegend=True,
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "deltaH0_vs_expected.png")

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaH0 predicted (unconstrained)"], 
    y=df_target_errors_unconstrained["deltaH0 rel. error"],
    name='Unconstrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaH0 predicted"], 
    y=df_target_errors["deltaH0 rel. error"],
    name='GHS constrained',
    mode='markers',
)

fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Rel. error vs Predicted values for delta H0",
    xaxis_title="Predicted delta H0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) H0 values",
    showlegend=True,
    font=dict(
        size=14,
    )
)

fig.show()
# -

# `deltaH0` residuals distribution:

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_target_errors_unconstrained["deltaH0 rel. error"], name='Unconstrained')
fig2 = go.Histogram(x=df_target_errors["deltaH0 rel. error"], name='GHS Constrained')

fig.add_traces([fig1, fig2])

fig.update_layout(
    xaxis_title="deltaH0 rel. error",
    barmode='overlay',
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.update_traces(opacity=0.75)

fig.show()
# -

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

fig_unconstrained = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["deltaG0 predicted (unconstrained)"],
    mode='markers',
    name='Predicted (unconstrained)'
)

fig.add_traces([fig_unconstrained, fig_predicted, fig_expected])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="deltaG0",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "deltaG0_scatter.png")

fig.show()
# -

# Against Expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaG0 expected"], 
    y=df_target_results["deltaG0 predicted"],
    name='GHS constrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaG0 expected"], 
    y=df_target_results["deltaG0 predicted (unconstrained)"],
    name='Unconstrained',
    mode='markers',
)

fig3 = go.Scatter(
    x=df_target_results["deltaG0 expected"], 
    y=df_target_results["deltaG0 expected"],
    name='Actual',
    mode='lines',
    line=dict(color="black", dash='dash'),
)

fig.add_traces([fig2, fig1, fig3])

fig.update_layout(
    title="Actual vs Predicted values for deltaG0",
    xaxis_title="Actual deltaG0 values",
    yaxis_title="Predicted deltaG0 values",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "deltaG0_vs_expected.png")

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["deltaG0 predicted (unconstrained)"], 
    y=df_target_errors_unconstrained["deltaG0 rel. error"],
    name='Unconstrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["deltaG0 predicted"], 
    y=df_target_errors["deltaG0 rel. error"],
    name='GHS constrained',
    mode='markers',
)

fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Rel. error vs Predicted values for delta G0",
    xaxis_title="Predicted delta G0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) G0 values",
    font=dict(
        size=14,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()
# -

# `deltaG0` residuals distribution:

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_target_errors_unconstrained["deltaG0 rel. error"], name='Unconstrained')
fig2 = go.Histogram(x=df_target_errors["deltaG0 rel. error"], name='GHS Constrained')

fig.add_traces([fig1, fig2])

fig.update_layout(
    xaxis_title="deltaG0 rel. error",
    barmode='overlay',
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.update_traces(opacity=0.75)

fig.show()
# -

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

fig_unconstrained = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["S0 predicted (unconstrained)"],
    mode='markers',
    name='Predicted (unconstrained)'
)

fig.add_traces([fig_unconstrained, fig_predicted, fig_expected])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="S0",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "S0_scatter.png")

fig.show()
# -

# Checking against expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["S0 expected"], 
    y=df_target_results["S0 predicted"],
    name='GHS constrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["S0 expected"], 
    y=df_target_results["S0 predicted (unconstrained)"],
    name='Unconstrained',
    mode='markers',
)

fig3 = go.Scatter(
    x=df_target_results["S0 expected"], 
    y=df_target_results["S0 expected"],
    name='Actual',
    mode='lines',
    line=dict(color="black", dash='dash'),
)

fig.add_traces([fig2, fig1, fig3])

fig.update_layout(
    title="Actual vs Predicted values for S0",
    xaxis_title="Actual S0 values",
    yaxis_title="Predicted S0 values",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "S0_vs_expected.png")

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["S0 predicted (unconstrained)"], 
    y=df_target_errors_unconstrained["S0 rel. error"],
    name='Unconstrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["S0 predicted"], 
    y=df_target_errors["S0 rel. error"],
    name='GHS constrained',
    mode='markers',
)

fig.add_traces([fig2, fig1])

fig.update_layout(
    title="Rel. error vs Predicted values for S0",
    xaxis_title="Predicted S0 values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) S0 values",
    font=dict(
        size=14,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()
# -

# `S0` residuals distribution:

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_target_errors_unconstrained["S0 rel. error"], name='Unconstrained')
fig2 = go.Histogram(x=df_target_errors["S0 rel. error"], name='GHS Constrained')

fig.add_traces([fig1, fig2])

fig.update_layout(
    xaxis_title="S0 rel. error",
    barmode='overlay',
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.update_traces(opacity=0.75)

fig.show()
# -

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

fig_unconstrained = go.Scatter(
    x=X_test["Molar Mass"], 
    y=df_target_results["Cp predicted (unconstrained)"],
    mode='markers',
    name='Predicted (unconstrained)'
)

fig.add_traces([fig_unconstrained, fig_predicted, fig_expected])

fig.update_layout(
    xaxis_title="Molar Mass",
    yaxis_title="Cp",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "Cp_scatter.png")

fig.show()
# -

# Checking against expected results:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["Cp expected"], 
    y=df_target_results["Cp predicted"],
    name='GHS constrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["Cp expected"], 
    y=df_target_results["Cp predicted (unconstrained)"],
    name='Unconstrained',
    mode='markers',
)

fig3 = go.Scatter(
    x=df_target_results["Cp expected"], 
    y=df_target_results["Cp expected"],
    name='Actual',
    mode='lines',
    line=dict(color="black", dash='dash'),
)

fig.add_traces([fig2, fig1, fig3])

fig.update_layout(
    title="Actual vs Predicted values for Cp",
    xaxis_title="Actual Cp values",
    yaxis_title="Predicted Cp values",
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "Cp_vs_expected.png")

fig.show()

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_target_results["Cp predicted (unconstrained)"], 
    y=df_target_errors_unconstrained["Cp rel. error"],
    name='Unconstrained',
    mode='markers',
)

fig2 = go.Scatter(
    x=df_target_results["Cp predicted"], 
    y=df_target_errors["Cp rel. error"],
    name='GHS constrained',
    mode='markers',
)

fig.add_traces([fig1, fig2])

fig.update_layout(
    title="Rel. error vs Predicted values for Cp",
    xaxis_title="Predicted Cp values",
    yaxis_title="Rel. error (actual - predicted) / (predicted) Cp values",
    font=dict(
        size=14,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()
# -

# `Cp` residuals distribution:

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_target_errors_unconstrained["Cp rel. error"], name='Unconstrained')
fig2 = go.Histogram(x=df_target_errors["Cp rel. error"], name='GHS Constrained')

fig.add_traces([fig1, fig2])

fig.update_layout(
    xaxis_title="Cp rel. error",
    barmode='overlay',
    font=dict(
        size=18,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.update_traces(opacity=0.75)

fig.show()
# -

# ### Verifying the consistency of the predicted values
#
# To evaluate the quality of the predictions from a thermodynamic point of view, we should assess the GHS residual to check if the predictions are consistent.

# * Thermodynamically constrained:

# +
df_predicted_species = df_nist_stdprops.loc[X_test_rescaled.index, ["Formula", "Molar Mass", "Se"]]
for target in list(df_y_predict.columns):
    df_predicted_species.loc[:, target] = df_y_predict.loc[:, target].values
    
df_predicted_species
# -

# * Thermodynamically unconstrained:

# +
df_predicted_species_unconstrained = df_nist_stdprops.loc[X_test_rescaled.index, ["Formula", "Molar Mass", "Se"]]
for target in list(df_y_predict_unconstrained.columns):
    df_predicted_species_unconstrained.loc[:, target] = df_y_predict_unconstrained.loc[:, target].values
    
df_predicted_species_unconstrained
# -

# Collecting the GHS residuals:

# +
T = 298.15  # in K
predicted_GHS_residuals = []
predicted_GHS_residuals_unconstrained = []
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
    G0_unconstrained = df_predicted_species_unconstrained.loc[index, "deltaG0"] * 1000
    H0_unconstrained = df_predicted_species_unconstrained.loc[index, "deltaH0"] * 1000
    S0_unconstrained = df_predicted_species_unconstrained.loc[index, "S0"]
    Se_unconstrained = df_predicted_species_unconstrained.loc[index, "Se"]
    GHS_residual_predicted_unconstrained = G0_unconstrained - H0_unconstrained + T * (S0_unconstrained - Se_unconstrained)
    predicted_GHS_residuals_unconstrained.append(GHS_residual_predicted_unconstrained)
    
    G0_expected = df_expected_stdprops.loc[index, "deltaG0"] * 1000
    H0_expected = df_expected_stdprops.loc[index, "deltaH0"] * 1000
    S0_expected = df_expected_stdprops.loc[index, "S0"]
    Se_expected = df_expected_stdprops.loc[index, "Se"]
    GHS_residual_expected = G0_expected - H0_expected + T * (S0_expected - Se_expected)
    expected_GHS_residuals.append(GHS_residual_expected)
    
df_predicted_species["GHS residual"] = predicted_GHS_residuals
df_predicted_species_unconstrained["GHS residual"] = predicted_GHS_residuals_unconstrained
df_expected_stdprops["GHS residual"] = expected_GHS_residuals
df_predicted_species
# -

# #### Visual verification

# * Thermodynamically constrained:

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

# * Thermodynamically unconstrained:

# +
fig = go.Figure()

fig1 = go.Scatter(
    x=df_expected_stdprops["GHS residual"], 
    y=df_predicted_species_unconstrained["GHS residual"],
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

# * Consistent vs inconsistent:

# +
fig = go.Figure()

fig1 = go.Histogram(x=df_expected_stdprops["GHS residual"], name='Expected GHS Residual')
fig2 = go.Histogram(x=df_predicted_species_unconstrained["GHS residual"], name='Unconstrained')
fig3 = go.Histogram(x=df_predicted_species["GHS residual"], name='GHS Constrained')

fig.add_traces([fig1, fig2, fig3])

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

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "GHS_residual_comparisons.png")

fig.show()
# -

# #### Quantitative verification
#
# We are able to compare the improvements of the predictions using GHS-constraints by comparing the mean and std deviations of the distributions related to the different approaches.

# * GHS-constrained:

# +
ghs_constrained_mean = df_predicted_species["GHS residual"].values.mean()
ghs_constrained_std_dev = df_predicted_species["GHS residual"].values.std()

print(f"Mean = {ghs_constrained_mean}\t Std-dev = {ghs_constrained_std_dev}")
# -

# * Unconstrained:

# +
unconstrained_mean = df_predicted_species_unconstrained["GHS residual"].values.mean()
unconstrained_std_dev = df_predicted_species_unconstrained["GHS residual"].values.std()

print(f"Mean = {unconstrained_mean}\t Std-dev = {unconstrained_std_dev}")
# -

# * Expected (from the database):

# Firstly, let's remove some outliers:

# +
quantile_cut = 0.015
quantile_low = df_expected_stdprops["GHS residual"].quantile(quantile_cut)
quantile_high  = df_expected_stdprops["GHS residual"].quantile(1 - quantile_cut)

df_expected_ghs_without_outliers = df_expected_stdprops[
    (df_expected_stdprops["GHS residual"] < quantile_high) & (df_expected_stdprops["GHS residual"] > quantile_low)
].copy()

df_expected_ghs_without_outliers
# -

# Actual values from the test set without outliers:

# +
expected_mean = df_expected_ghs_without_outliers["GHS residual"].values.mean()
expected_std_dev = df_expected_ghs_without_outliers["GHS residual"].values.std()

print(f"Mean = {expected_mean}\t Std-dev = {expected_std_dev}")
# -

# Visual comparison:

# +
stats_labels = ['GHS constrained', 'Unconstrained', 'Expected']

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add the mean values bar chart
fig.add_trace(
    go.Bar(
        x=stats_labels,
        y=np.abs([ghs_constrained_mean, unconstrained_mean, expected_mean]),
        text=[f'{x:.1f}' for x in np.abs([ghs_constrained_mean, unconstrained_mean, expected_mean])],
        textposition='outside',
        name="GHS residual absolute mean value",
        textfont=dict(size=16),
        offsetgroup=1,
    ),
    secondary_y=False,  # This trace goes on the primary y-axis
)

# Add the std. dev. bar chart
fig.add_trace(
    go.Bar(
        x=stats_labels,
        y=[ghs_constrained_std_dev, unconstrained_std_dev, expected_std_dev],
        text=[f'{x:.1f}' for x in [ghs_constrained_std_dev, unconstrained_std_dev, expected_std_dev]],
        textposition='outside',
        name="GHS residual std. dev.",
        textfont=dict(size=16),
        offsetgroup=2,
    ),
    secondary_y=True,  # This trace goes on the secondary y-axis
)

# Set the number of ticks you want
num_ticks = 8

# Calculate tick values for the primary y-axis
max_primary = max(np.abs([ghs_constrained_mean, unconstrained_mean, expected_mean]))
primary_tickvals = np.linspace(0, max_primary, num_ticks)

# Calculate tick values for the secondary y-axis
max_secondary = max([ghs_constrained_std_dev, unconstrained_std_dev, expected_std_dev])
secondary_tickvals = np.linspace(0, max_secondary, num_ticks)

# Set y-axes titles
fig.update_yaxes(
    title_text="GHS residual absolute mean value",
    secondary_y=False,
    tickvals=primary_tickvals,
    range=[0, max_primary * 1.1],
)
fig.update_yaxes(
    title_text="GHS residual std. dev.",
    secondary_y=True,
    tickvals=secondary_tickvals,
    range=[0, max_secondary * 1.1]
)

# Calculate new x-values for the second trace to offset the bars
offset = 0.0  # This value might need to be adjusted
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        # Position the tick in the middle of the grouped bars
        tickvals=[i + offset/2 for i in range(len(stats_labels))],
        ticktext=stats_labels
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.15
    ),
    font=dict(
        size=18,
    ),
    bargap=0.1
)

if is_saving_figure_enabled:
    fig.write_image(RESULTS_PATH / "GHS_residuals_means_and_stddev.png")

fig.show()
