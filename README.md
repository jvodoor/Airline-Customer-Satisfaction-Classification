DATA 4950 Capstone Project
==============================

This Project will analyze various data points surrounding the customer satisfaction of an airline customer. Based on the types of flights and customer, can we predict who is more likely to be a satisfied customer and derive insights for airline companies?

The raw dataset contains ~130,000 surveys taken by airline customers from an anonymous airline. Each passenger is sorted by their type of flying class (economy, economy plus, business), their gender, and age. Each passenger also has flight data from the flight they flew, representing the distance, as well as any delays on departure or arrival. The passenger then fills out a survey of various factors of the entire airline experience, from purchasing the ticket, to inflight amenities such as wifi service, and all other aspects of customer service. In total the customer was asked to rate the service received from 14 different aspects of their flight from a scale of 1-5, and then their final satisfaction level of satisfied or not satisfied. 

Using this data, we want to determine whether we can predict the satisfaction of a customer, and determine which are the most important aspects of service that the airline should use in order to maximize customer satisfaction. Given the wide variety of customer facing aspects that affect an airline business, it's important that we try to narrow down the most important features that an airline can focus on, as prioritizing certain aspects of the business is a much more accomplishable goal than to broadly focus on all aspects of the business. 

The project starts off by using logistic regression to predict how likely we can classify a passenger of being satisfied or not based on the available survey data we have, and then narrow down the list of features to determine whether we can make a more accurate model, or at least a model of similar accuracy to be better explainable to the airline. 

The project then uses an alternative model to determine if better accuracy and insights can be gained. The alternate model has not been set yet, but this will most likely be a neural net. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
