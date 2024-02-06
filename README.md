# finmetrika-ml
Customized machine learning library for practical projects.

# Installation
Install with
```bash
pip install finmetrika-ml
```

# Finmetrika ML Project Structure
```
finmetrika_ml
├── data
    ├── processing.py
    └── vizualization.py
├── model
    ├── training.py
    ├── evaluation.py
    └── metrics.py
└── utils.py
```

- `data/` directory contains scripts that aid to process, analyze and visuzalize data. The processing also includes encoding of input data to be used for deep learning training.
    - `processing.py`: preprocessing and processing scripts for input data
    - `vizualization.py`: create insightful plots and charts for data analysis and results presentation.
- `model/`: directory containing scripts for model training, evaluation and metrics.
    - `training.py`: Scripts and notebooks for training machine learning models. This also includes fine tuning LLMs.
    - `evaluation.py`: Scripts and notebooks dedicated to evaluating the performance of the trained models.
    `metrics.py`: Modules for calculating and storing metrics related to model performance.
- `utils.py`: Utility functions used across the project for various tasks such as logging, writing experiment information, checking the compute device, reproducibility, etc.

# Examples


# Documentation
See detailed documentation in [documentation]().
