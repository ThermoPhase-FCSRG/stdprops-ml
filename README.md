# Std. thermodynamic properties estimation with machine learning

This repo contains studies of how to explore, analyze, and model (with Multilayer Percepton neural network, or simply MLP) std. state data publicly available and provided by [NIST (NBS tables)](https://data.nist.gov/od/id/mds2-2124).

## Goals

In typical geochemical routines, chemical equibilibirum and kinetics are frequent calculations. To run such simulations, two main ingredients are required:

* A thermodynamic database containing std. state (reference state) of properties like the Gibbs free energy of formation, Enthalpy of formation, Entropy of formation, and heat capacities. These properties are necessary to compute the ideal part of chemical potentials, which are used to calculate equilibrium or kinetics for a given system.
* Activity models and Equations of State to compute the excess (non-ideal) part of the chemical potentials.

The goal of this study is to provide and evaluate different strategies to predict the std. state properties that are new (or lacking) to a given database. Such scenarios are not uncommon since geochemical problems and systems usually have a lot of different chemical species in different phases (gas, aqueous, minerals, etc), posing a challenge to have all the std. state properties to fully model the systems.

## The studies

A set of MLP approaches exploring different features are provided in `notebooks`. The most complete version is in `notebooks/stdprops-ml-predictions (MM, Charge, State, Se, Elements).ipynb`, which the molar mass, charge, state of matter, entropy of the elements in the species chemical formulas (which is given by tables, see [this table from CHNOSZ, for example](https://github.com/jedick/CHNOSZ/blob/main/inst/extdata/thermo/element.csv)), and the number of elements in the species are used as featured to predict the std. properties at $P = 1$ bar and $T = 25$ degC. Simplified approaches are also provided to demonstrate the limitations and how the features are relevant to improve the predictions. Exploratory Data Analysis (EDA) for the NIST/NBS tables is also performed in `notebooks/nist-exploratory-data-analysis.ipynb` to aid insights and to check the consistency of this database.

## Roadmap

The main milestones of this research are the following:

* [X] EDA of NIST/NBS table.
* [X] ML modeling and predictions using Molar Mass and Species Charges as features.
* [X] ML modeling and predictions using Molar Mass, Species Charges, and State of Matter as features.
* [X] ML modeling and predictions using Molar Mass, Species Charges, State of Matter, Entropy of Elements, and Number of Elements as features.
* [ ] Consistent ML modeling and predictions using Molar Mass, Species Charges, State of Matter, Entropy of Elements, and Number of Elements as features. In this approach, the G-H-S relationship (as described in the [NIST/NBS tables original report](https://srd.nist.gov/JPCRD/jpcrdS2Vol11.pdf)) is incorporated in the loss function, resulting in thermodynamically consistent predictions.
* [ ] (WIP) Create a reproducible environment to run the studies and record it in Zenodo.
* [ ] Write a manuscript (preprint) about the research.
* [ ] Submit the manuscript to a related Journal.

## Contact

My name is Diego. If you have interest in this research or want to discuss something, please open an issue or reach me out (contact info provided in my github profile).
