# ACE-BCC-TMs

This repository collects atomistic modeling inputs, fitted ACE potentials, and analysis scripts for body-centered-cubic (bcc) transition metals, with a focus on screw dislocations and line-tension modeling.

The current materials covered here are:

- V
- Nb
- Ta
- Mo
- W

## Repository layout

- `ACE2020_potential/`: ACE potential inputs, fitted outputs, and database files for the 2020 potential set.
- `ACE2025_potential/`: ACE potential inputs, fitted outputs, and database files for the 2025 potential set.
- `BCC_Screw_Dislocation/`: Python scripts for building and analyzing bcc screw-dislocation configurations.
- `VASP_inp_expample/`: example VASP input templates and helper scripts for collecting and post-processing calculation outputs.
- `LT_model/`: a marimo-based notebook app and exported HTML for exploring the line-tension model and plotting predicted kink-pair formation energies.

## Main contents

### ACE potential folders

Both `ACE2020_potential/` and `ACE2025_potential/` are organized by element. Typical contents include:

- `input.yaml`: fitting input used to generate the ACE potential
- `output_potential.yaml`: fitted potential in YAML form
- `output_potential.asi`: exported potential file
- `Database/*.pckl.gzip`: compressed data files used in fitting or analysis

### Screw dislocation scripts

The `BCC_Screw_Dislocation/` folder contains Python tools for constructing and analyzing bcc screw dislocations.

### VASP examples

The `VASP_inp_expample/` folder provides example workflows for primitive-cell calculations and output collection, including template `INCAR` and `POTCAR` files and helper shell/Python scripts.

### Line-tension model

The `LT_model/` folder contains:

- `LT_model.py`: the editable marimo notebook source
- `LT_model.html`: a shareable HTML export of the notebook
- `README.md`: notes on the LT-model workflow and how marimo is used in this project

For more detail, see [LT_model/README.md](/Users/leizhang/Nextcloud/G_github/ACE-BCC-TMs/LT_model/README.md).

## Using the repository

Most analysis scripts are written in Python. The exact dependencies vary by workflow, but common packages used in this repository include:

```bash
pip install numpy scipy matplotlib marimo
```

To work with the line-tension notebook:

```bash
cd LT_model
marimo edit LT_model.py
```

To run it directly as a Python app:

```bash
python LT_model.py
```

## Notes

- Generated marimo session files under `__marimo__/` are local runtime artifacts and should not be committed.
- Large fitting outputs and database files are kept in the repository because they are part of the scientific workflow and reference data.

## License

This repository is distributed under the terms of the license in [LICENSE](/Users/leizhang/Nextcloud/G_github/ACE-BCC-TMs/LICENSE).
