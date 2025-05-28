# Temperature-in-beehives
This repository contains the implementation of the model and data analysis of two methods proposed in Ref. [1] for diagnosing the status of a honeybee hive using time series data of hive and environmental temperatures.  


## Usage

The jupyter notebooks in the root file of the repository allow you to explore the results:
- Tutorial_stable_hive: Simple tutorial to explore the two metodologies to calculate $\Pi$ and $\Delta T$ for a hive that performs well 
- Tutorial_unstable_hive: Simple tutorial to explore the two metodologies to calculate $\Pi$ and $\Delta T$ for a hive that eventually collapses. 
- Computational_methods*: Computes results for each datasets and reproduces the results presented in the manuscript.
- Results*: Reproduces the figures presented in the manuscript, using the output from the computational methods file.

## Folders

- src: source codes, in the file "source_codes.py"
- data: 22 data sets of temperature in honey bee hives divided into two data sets
- ouput: pickles and figures generated from running the codes


## Reference
[1] Arias-Calluari, K., Colin, T., Latty, T., Myerscough M., & Altmann, E. G. "Assessing Honey Bee Colony Health Using Temperature Time Series"
