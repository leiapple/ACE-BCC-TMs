# LT Model

This folder contains a small [marimo](https://marimo.io/) notebook app for exploring the line-tension (LT) model used to predict kink-pair formation energies in bcc transition metals.

## Files

- `LT_model.py`: the source notebook as a marimo app. It defines the LT-model helper functions, solves for `R` and `alpha` from MD-derived inputs, and plots the predicted `ΔH*(τ)` curves for selected elements.
- `LT_model.html`: an exported standalone HTML view of the notebook output. This is useful for sharing results without requiring Python or marimo.
- `__marimo__/`: marimo runtime metadata created while editing or running the notebook locally. This folder is machine-generated and is ignored by git.

## Why marimo is used here

marimo is a Python notebook environment that stores notebooks as plain `.py` files instead of JSON documents. That makes it a good fit for this repository because:

- the notebook source is readable and version-controllable,
- diffs are much cleaner than traditional notebook files,
- code, markdown, and plots stay together in one executable script,
- the HTML export can be shared separately from the editable source.

## Running the notebook

Create or activate a Python environment with the required packages:

```bash
pip install marimo numpy scipy matplotlib
```

Then start the notebook app from this directory:

```bash
marimo edit LT_model.py
```

If you just want to run it as a script:

```bash
python LT_model.py
```

## Workflow

1. Edit `LT_model.py` in marimo.
2. Re-run the cells to update plots and derived quantities.
3. Export or refresh `LT_model.html` when you want a shareable snapshot of the current notebook output.
4. Do not commit `__marimo__/` contents; they are local session artifacts.
