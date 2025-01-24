import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

def get_model_for_depths(depth_range=range(1,8)):
    models = dict()
    depths = [i for i in depth_range] + [None]
    for n in depths:
        models[str(n)] = RandomForestClassifier(max_depth=n)

    return models

def get_models_samples_ratio(samples_ratio_range=np.arange(0.1, 1.1, 0.1)):
    models = dict()
    for i in samples_ratio_range:
        key = '%.1f' % i
        if i == 1.0:
            i = None
        models[key] = RandomForestClassifier(max_samples=i)
    return models

def get_models_number_of_features(num_features=range(1, 8)):
    models = dict()
    for i in num_features:
        models[str(i)] = RandomForestClassifier(max_features=i)
    return models

def get_models_for_trees(n_estimators=[10, 50, 100, 500, 1000]):
    models = dict()
    for n in n_estimators:
        models[str(n)] = RandomForestClassifier(n_estimators=n)
    return models

def evaluate_model(model, X, y, n_splits=10, n_repeats=3):
    # define the evaluation process
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    # evaluate the model and collect results
    scores = cross_val_score(model, X, y, scoring="f1_weighted", cv=cv, n_jobs=-1)
    return scores

def plot_feature_importances(importances, feature_list, num_features="all", return_importances=False, print_importances=False, figsize=(22, 6), save_path=None):
    matplotlib.rcParams['font.size'] = 50
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances
    if print_importances:
        for pair in feature_importances:
            print('Variable: {:50} Importance: {}'.format(*pair))
    if num_features == "all":
        num_features = len(feature_importances)
    plt.figure(figsize=figsize)
    # %matplotlib inline # Set the style
    plt.style.use('fivethirtyeight')# list of x locations for plotting
    x_values = list(range(len(importances)))# Make a bar chart
    plt.bar(x_values[:num_features], list(pd.DataFrame(feature_importances)[1].values[:num_features]), orientation = 'vertical')# Tick labels for x axis
    plt.xticks(x_values[:num_features], list(pd.DataFrame(feature_importances)[0].values[:num_features]), rotation=30, ha="right")#'vertical')# Axis labels and title
    plt.ylabel('Importance') 
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    if save_path is not None:
        if not(os.path.exists(save_path)):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, "feature_importances_rf.png"), bbox_inches='tight')
                    
    if return_importances:
        return feature_importances
