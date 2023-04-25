DATA 4950 Capstone Project
==============================

This Project analyzes various data points surrounding the customer satisfaction of an airline customer. Using customer survey data, can we predict who is more likely to be a satisfied customer and derive insights for airline companies?

The raw dataset contains ~130,000 surveys taken by airline customers from an anonymous airline. Each passenger is sorted by their type of flying class (economy, economy plus, business), their gender, and age. Each passenger also has flight data from the flight they flew, representing the distance, as well as any delays on departure or arrival. The passenger then fills out a survey of several factors of the entire airline experience, from purchasing the ticket, to in-flight amenities such as Wi-Fi service, and all other aspects of the customer experience. In total the customer was asked to rate the service received from 14 distinct aspects of their flight from a scale of 1-5, and then their final satisfaction level of satisfied or not satisfied.

Using this data, we want to determine whether we can predict the satisfaction of a customer and determine which are the most important aspects of service that the airline should use in order to maximize customer satisfaction. Given the wide variety of customer facing aspects that affect an airline business, it is important that we try to narrow down the most notable features that an airline can focus on, as prioritizing certain aspects of the business is a much more accomplishable goal than to broadly focus on all aspects of the business.

The project starts off by using logistic regression to predict the probability of classifying a passenger of being satisfied or not based on the available survey data we have. We then use an alternative Decision Tree model called a XGBoosted Tree, which allows us to construct a tree of choices with various predictors to determine the most accurate way to classify customer satisfaction. Lastly we explore additional models featuring the removal of Gender as a variable to avoid Gender bias, and then narrow down the list of features to the top ten per model to have a more explainable model. An alternative model approach was considered using a neural network, but the performance scores were too weak compared to other models as it stands. The neural network’s score likely could be improved with a number of complex tweaks to its architecture.

The logistic regression model is one of the most simplistic and interpretable models. It gives an easily explainable probability metric of whether a customer will be satisfied based on how they respond to the survey data. A XGBoosted Tree model is also an easily explained concept, quite possibly the easiest of all models to explain. You choose a determining factor for a predictor, and then analyze further predictors “down the line” until you arrive at an appropriate group. For example, if the customer scored “Inflight Service” as greater than 2.5, it would go down to the right side of the tree. If they scored service as less than 2.5, it would go down the left side of the tree. Repeat with a grouping of predictors until the tree reaches our specified tree height of five, and an ending leaf value represents how much or less likely a customer is to be satisfied. 

Examining the strength of our models, both performed admirably. After analyzing and determining the most important predictors that should stay in our model, we finished training the model under the above-mentioned algorithms, and then conducted model predictions based on test data it had not seen before. We then calculate a variety of scores to indicate the strength of the models. Accuracy, and ROC AUC Score were used.

Accuracy represents the percentage of our model predicting correctly whether a customer is satisfied or not. This is done by the model making a prediction, and then checking that prediction against the target output of a row of customer survey data. It then repeats the process over an entire dataset to determine percentages. Our Logistic Regression model has a strong correct prediction rate of 87.6%, while our XGBoosted Tree model has an even stronger performance of 94.6%. Our other metric is ROC AUC Score. The ROC AUC score is represented in a graph, and it examines the relationship as you increase the number of times the model will classify customers as being satisfied. At the end of the curve, the model suggests all customers are satisfied, whereas at the beginning the model suggests all customers are not satisfied. The goal is to draw a slope line from the beginning to end of the curve and determine the point farthest away from the slope. This is our ROC Score, where 1 is the highest. For our logistic regression model, the ROC score was a strong 94.5%, whereas for our XGBoosted Tree model, it was an even stronger 98.9%. This means we can be extremely confident in both models’ ability to generate correct customer satisfaction predictions, with the XG Boosted Tree model leading the way.

Drawing more in-depth assessments, it is quite interesting that the two models diverged quite strongly on which predictors are most important. The logistic model identified the type of travel a customer made, whether business or pleasure, as being the most crucial factor to predict satisfaction. If a customer was a business traveler, they were 2.6 times more likely to be satisfied over a non-business traveler. On the other hand with the XGBoosted Tree model, “Online boarding,” the process of boarding the plane, was the most important coefficient, with the predictor accounting for over 35% of the total importance alone. Fittingly, both models take the number 1 of the other model and have them as a strong number 2 in terms of importance. We can be fairly confident then that an airline company should prioritize smoothing the boarding process, and ensuring business customers continue to remain satisfied.  As we move into more targeted survey metrics, Inflight Wi-Fi and entertainment appear to be strong key factors to a customer being satisfied. These are no doubt expensive endeavors for companies to implement, but with how popular they remain with passengers, it will be a positive experience. There are some divergences, as the logistic regression model believes convenient departure and arrival times offer downward pressure on the probability rate of satisfaction interestingly enough, whereas the tree model does not consider it important. We could make an inference that convenient times do not necessarily matter for customers, but this is a bold assumption that would necessitate more tests in order to come to this conclusion. 

Overall, it would appear that with the exception of the boarding process, improving the in-flight product in a variety of ways seems to be the most crucial factor in increasing customer satisfaction. Inflight Wi-Fi service, entertainment, leg room, and on-board service all showed up as important predictors in both models. While improving the ground product is important as the boarding scores show, if an airline is able to differentiate itself with a superior in-flight product, they may be able to have increased customer satisfaction, and as a result, this may lend itself to increased revenue. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    ├── models             <- Trained and serialized models
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
            └── alt_neural_net.py <- alternate model approach that was not used as final comparison, but could
                                    be improved later  
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
