# Goal is to print the hyperparameters of the computation pipeline

import computation_pipeline_hyper_params as params

# Get a list of all names in the module
names = dir(params)

# Print each name and its value
for name in names:
    if not name.startswith("__"):  # Ignore special methods
        value = getattr(params, name)
        print(f"{name} = {value}")
