# Task 2 — Decision Trees

Starter files for **Task 2** (graded, 25 points, due Friday 9 October 2026, 23:59 Canvas time, UTC+5)
of *Machine Learning (Python)*, MOP 3231, Narxoz University.

The task text, the five parts, the grading table and the submission rules are in the handout in the
**Week 05** module in Canvas. This repository holds only the files you start from; it has no
instructions of its own beyond the setup below.

## What is here

| File | What it is |
|---|---|
| `data/heart_disease.csv` | The dataset: 297 patients, 13 features and the target `disease` |
| `requirements.txt` | Pinned library versions the task is written and checked against |
| `make_dataset.py` | How `heart_disease.csv` was built from the UCI source. You do not need to run it |
| `.gitignore` | Keeps your virtual environment and notebook checkpoints out of git |

## How to start

You do not push to this repository. You make **your own** repository, as in Task 1, and submit the
URL of its final commit in Canvas.

1. Download this repository as a ZIP (**Code → Download ZIP**) and unzip it into a new folder named
   `lastname_ML_task_2`.
2. Create the virtual environment and install the pinned libraries:

   ```bash
   cd lastname_ML_task_2
   python3 -m venv .lastname_ML_task_2
   source .lastname_ML_task_2/bin/activate      # Windows PowerShell: .lastname_ML_task_2\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Work in a notebook called `lastname_task2.ipynb` in the root of the folder, next to `data/`.
4. `git init`, commit as you go, push to your own GitHub repository and submit the final commit URL.
   The exact steps and what must be in the repository are in the handout.

## The data

`data/heart_disease.csv` is the Cleveland subset of the **Heart Disease** dataset from the UCI Machine
Learning Repository, licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C52P4X.

Original paper: Detrano, R. et al. (1989), *International application of a new probability algorithm
for the diagnosis of coronary artery disease*, American Journal of Cardiology 64:304–310.

**Changes made to the original file** (`processed.cleveland.data`): the 6 of 303 rows with a missing
value (in `ca` or `thal`) were dropped; the 0–4 diagnosis column `num` was replaced by the binary
`disease` (1 if `num > 0`); integer-valued columns were stored as integers; columns were named after the
UCI attribute names. Nothing else was changed.

This is a teaching dataset, not clinical evidence. Do not draw medical conclusions from a model trained on it.
