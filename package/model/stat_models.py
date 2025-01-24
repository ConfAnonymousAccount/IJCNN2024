import pathlib
from typing import Union

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

from package.data.scaler import StandardScaler
from package.data.data_manager import DataManager
from package.model.utils import compute_metrics
from package.data.utils import round_off_rating

class StatisticalModels(object):
    def __init__(self,
                 model,
                 config_path: Union[pathlib.Path, str],
                 config_name: str="DEFAULT",
                 name: str="test_model",
                 scaler: Union[StandardScaler, None]=None,
                 indices_to_scale: Union[list, None]=None,
                 seed: int=42,
                 **kwargs):
        self.trained = False
        self.params = kwargs
        self.model = model
        self._model = self.model(name=name, 
                                 config_path=config_path, 
                                 config_name=config_name, 
                                 scaler=scaler,
                                 indices_to_scale=indices_to_scale,
                                 seed=seed, 
                                 **kwargs)
        self.params.update(self._model.params)
        config_name = self._model.config.section_name
        self.name = name + '_' + config_name

        self._observations = dict()
        self._predictions = dict()

        # history
        self.train_metrics = {}

    def build_model(self):
        self._model.build_model()
        self.trained = False

    def train(self,
              train_dataset,
              save_path: Union[None, str]=None,
              scale_only_target: bool=False,
              **kwargs):
        self.params.update(kwargs)

        features, targets = self._model.process_dataset(train_dataset, training=True, scale_only_target=scale_only_target)

        # build the model
        self.build_model()

        # fit the model
        self._model.model.fit(features, targets)

        self.trained = True
        if save_path is not None:
            self.save(save_path)

    def predict(self, dataset, **kwargs):
        if self.trained == False:
            raise Exception("The model is not yet trained! Try to call train function before calling predict!")
        test_features, test_targets = self._model.process_dataset(dataset, training=False)

        predictions = self._model.model.predict(test_features)
        predictions = self._model._post_process(predictions)
        targets = self._model._post_process(test_targets)

        self._predictions = predictions
        self._observations = targets

        return predictions
    
    def cross_validate(self, dataset, n_splits: int=5, random_state: int=123, shuffle: bool=True, verbose: int=1, round_out: bool=True):
        cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        models, predictions, true_labels = list(), list(), list()
        metrics = []

        for (train, test), i in tqdm(zip(cv.split(dataset.features, dataset.targets), range(n_splits)), total=n_splits):
            data_manager = DataManager()
            data_manager.split_by_index(dataset, train_indices=train, test_indices=test)
            self.build_model()
            self.train(data_manager.train_dataset)
            models.append(self._model.model)
            prediction = self.predict(data_manager.test_dataset)
            if round_out:
                observations = np.array([round_off_rating(el) for el in data_manager.test_dataset.targets.ravel()])
                prediction = np.array([round_off_rating(el) for el in prediction])
            else:
                observations = data_manager.test_dataset.targets.ravel()
            metric = compute_metrics(observations, prediction, index=-1)
            predictions.append(prediction)
            true_labels.append(observations)
            metrics.append(metric)

        all_metrics = dict()
        metric_list = metric.keys()
        for metric in metric_list:
            all_metrics[metric] = []
            for i in range(n_splits):
                all_metrics[metric].append(metrics[i][metric])
            if verbose == 1:
                print(f"{metric} : Mean = {np.mean(all_metrics[metric]):.2f}, Std = {np.std(all_metrics[metric]):.2f}")

        return all_metrics, models, predictions, true_labels