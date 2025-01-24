from typing import Union
import copy
import pathlib
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from operator import itemgetter 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from package.model.torch_models import TorchModels
from package.data.dataset import DataSet
from package.data.utils import round_off_rating

def compute_metrics(true, pred, index=0):
    if index >= 0:
        mae = mean_absolute_error(true[index], pred[index])
        mse = mean_squared_error(true[index], pred[index])
        mape = mean_absolute_percentage_error(true[index], pred[index])
        pearson = pearsonr(true[index], pred[index])
        spearman = spearmanr(true[index], pred[index])
        #r, prob = pearson
    else:
        if type(true) == list:
            true = np.concatenate(true)
        if type(pred) == list:
            pred = np.concatenate(pred)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        pearson = pearsonr(true.flatten(), pred.flatten())
        spearman = spearmanr(true.flatten(), pred.flatten())
    metric_dict = {}
    metric_dict["mae"] = mae
    metric_dict["mse"] = mse
    metric_dict["mape"] = mape
    metric_dict["pearson"] = pearson
    metric_dict["spearman"] = spearman

    return metric_dict

def k_fold(features, targets, folds):
    data_len = len(features)
    train_datasets = []
    test_datasets = []

    kf = KFold(n_splits=folds)
    print("number of splits:", kf.get_n_splits(np.arange(data_len)))

    for i, (train_index, test_index) in enumerate(kf.split(np.arange(data_len))):
        sample_features_train = np.take(features, train_index, axis=0)
        sample_targets_train = np.take(targets, train_index, axis=0)
        train_dataset = DataSet()
        train_dataset.set_features(sample_features_train)
        train_dataset.set_targets(sample_targets_train)
        sample_features_test = np.take(features, test_index, axis=0)
        sample_targets_test = np.take(targets, test_index, axis=0)
        test_dataset = DataSet()
        test_dataset.set_features(sample_features_test)
        test_dataset.set_targets(sample_targets_test)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    return train_datasets, test_datasets

def cross_validation(model, 
                     train_datasets: list, 
                     test_datasets: list, 
                     config_path: Union[pathlib.Path, str], 
                     config_name: str, 
                     model_name: str="FC",
                     scaler=None,
                     indices_to_scale=None,
                     scale_only_features=True):
    model_name = model_name + '_' + config_name
    trained_models = []
    metrics_all = []
    for i in range(len(train_datasets)):
        
        model_cv = TorchModels(model,
                               config_path=config_path,
                               config_name=config_name,
                               name="fully_connected",
                               scaler=scaler,
                               indices_to_scale=indices_to_scale
                              )
        model_cv.train(train_dataset=train_datasets[i], scale_only_features=scale_only_features)
        predictions = model_cv.predict(test_datasets[i], scale_only_features=scale_only_features)
        predictions = np.array([round_off_rating(el) for el in predictions.ravel()])
        targets = np.array([round_off_rating(el) for el in test_datasets[i].targets.ravel()])
        metrics = compute_metrics(targets, predictions, index=-1)
        # metrics = compute_metrics(test_datasets[i].targets, predictions.round(), index=-1)
        trained_models.append(model_cv)        
        metrics_all.append(metrics)
        # pprint(metrics)
    return trained_models, metrics_all
