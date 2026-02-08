# Third-Party Notices

This repository depends on third-party open-source software.
No third-party source trees are vendored in this cleaned release; dependencies are installed via `pip`.

## Core Dependency
- **BOP Toolkit**
  - Upstream: https://github.com/thodan/bop_toolkit
  - Pinned revision: `a9c7ae92a97a9f57e684ea2a995b8c61da3b16c1`
  - Usage: BOP dataset parameter resolution and evaluation utilities
  - License: See upstream repository license files

## Optional Evaluation Dependencies
- `torch-fidelity` (FID/KID/PR metrics)
- `clean-fid`
- `pycocotools`
- `fvcore`
- `ptflops`
- `ultralytics` (for `evaluation/yolo` pipeline)

Please refer to each package's upstream project for exact license terms.

## Data and Checkpoints
- Datasets and model checkpoints are **not** redistributed in this repository.
- Users are responsible for complying with original dataset/checkpoint licenses.
