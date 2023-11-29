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

# # Chemical Formula parser
#
# Convenient function to get the amount of species in a Chemical Formula. Below a bunch of test cases are provided to verify the implementation of the function.

# +
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


selected_test_formulas = [
    "NaCl", "H2O", "C6H12O6", "NH4NO3", "(NH4)2SO4", "CH3COOH", "C2H5OH", 
    "C12H22O11", "NaHCO3", "C8H18", "Ca(OH)2", "Al2(SO4)3", "K2Cr2O7", 
    "KMnO4", "FeS2", "Mg(OH)2", "Na2CO3·10H2O", "CuSO4·5H2O", "(NH4)3PO4", 
    "H2SO4", "Na<sup>+</sup>", "C6H5OH", "C18H27NO3", "(CaO)2·MgO·2SiO2",
    "Fe2O3", "Pb(NO3)2", "MgSO4·7H2O", "Ca3(PO4)2", "H3PO4", "C2H3NaO2",
    "KAl(SO4)2·12H2O"
]

# Run the parser on these selected formulas
selected_parsed_results = {
    formula: parse_chemical_formula(formula) for formula in selected_test_formulas
}
selected_parsed_results
