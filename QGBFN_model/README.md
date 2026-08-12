# 1.5 layer Quasi-geostrofic model w. Back-andForward nuding data assimilation 

This folder contains scripts used to make the QG model run. With how all files was made and the code finally run 

## Contents

```text
QGBFN_model/
├── c0_process.py              # Code used to load c0 and plot c0 and L_R.
├── make_nk_landmask.py        # Code used to make the landmask for model, using a Norkyst datafile
├── make_NK_BC.py              # Code used to make the Norkyst daily average ADT DAC files for inital concition and boundary condition.
├── make_file.py               # Code used to make the SWOT vs NK vs QG model files.
├── comparison_1_SWOTvNKvQG.py # Code used to make the SWOT vs NK vs QG comparisons and plots.
├── scr.zip                    # zip file containing load of code for model lighlty changed for c0 variablity, else made by Le Guillou et. al 2023 and lighlty edited by  Jensen, 2025.
├── models.zip                 # zip file containing code for model made by Le Guillou et. al 2023
└── README.md
```

Sources:

Florian Le Guillou, Renaud-Matthias, Sammy Metref& Juan Emmanuel Johnson. (2023). leguillf/MASSH: New release (Version v2.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.10017533

Jensen, S. N., Andersen, O. B., Ludwigsen, C. B., Gonçalves‐Araujo, R., & de Steur, L. (2025). Surface water and ocean topography (SWOT) observations unveil small mesoscale variability on the East Greenland shelf. Geophysical Research Letters, 52, e2025GL118573. https://doi.org/10.1029/2025GL118573
