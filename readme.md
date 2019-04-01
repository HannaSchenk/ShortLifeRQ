#  Simulation code for “The short life of the Red Queen”

This repository contains code, numerical results and stochastic simulation results for generating all Figures in the manuscript.

**Easy start:** Have a look at the Jupyter notebook Trajectory_Fig1.ipynb which explains and lets you try out the simulations.

* The “main” scripts perform the stochastic simulations for two or more types: “twoTypes” can also generate Fig1, “manyTypes” generates Fig4 and Fig5

* The “plot_from_data” script can generate Fig2 and Fig3 from data stored in data_main and data_w.

* Other scripts contain mathematical details and algorithms (see Supplementary material of the publication) and are called by the main scripts.

* The “sojourn” text files contain exact mean extinction times calculated with the Mathematica 10.2 notebook and are compared to simulation results in Fig3