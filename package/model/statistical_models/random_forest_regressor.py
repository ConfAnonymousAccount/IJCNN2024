import os
import pathlib
from typing import Union
from copy import deepcopy

from sklearn.ensemble import RandomForestRegressor

from package.config import ConfigManager
from package.data.scaler import StandardScaler

class StatRandomForestRegressor(object):
    def __init__(self,
                config_path: Union[pathlib.Path, str],
                config_name: Union[pathlib.Path, str],
                name: str="test",
                scaler: Union[StandardScaler, None]=None,
                indices_to_scale: Union[list, None]=None,
                seed=42,
                **kwargs):
        
        if not os.path.exists(config_path):
            raise RuntimeError("Configuration path not found!")
        self.model = None
        config_name = config_name if config_name is not None else "DEFAULT"
        self.config = ConfigManager(section_name=config_name, path=config_path)
        self.name = name if name is not None else self.config.get_option("name")
        self.scaler = scaler
        self.indices_to_scale = indices_to_scale
        if self.scaler is not None:
            self.scaler = scaler()

        self.params = self.config.get_options_dict()
        self.params.update(kwargs)

        self.seed = seed

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]
        self.data_size = None

    def build_model(self):
        self.model = RandomForestRegressor(n_estimators=self.params["n_estimator"], 
                                           max_depth=self.params["max_depth"], 
                                           max_samples=self.params["max_samples"], 
                                           max_features=self.params["max_features"], 
                                           random_state=self.seed)

    def process_dataset(self, dataset, training: bool, scale_only_target: bool=False):
        if training:
            extract_x = deepcopy(dataset.get_features())
            extract_y = deepcopy(dataset.get_targets())
            self._infer_size(extract_x, extract_y)
            if self.scaler is not None:
                self.scaler.fit(extract_x, extract_y, self.indices_to_scale)
                #extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
                extract_y = self.scaler.transform_y(extract_y)
                if not(scale_only_target):
                    extract_x = self.scaler.transform_x(extract_x)
        else:
            extract_x = deepcopy(dataset.get_features())
            extract_y = deepcopy(dataset.get_targets())
            self._infer_size(extract_x, extract_y)
            if self.scaler is not None:
                #extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
                extract_y = self.scaler.transform_y(extract_y)
                if not(scale_only_target):
                    extract_x = self.scaler.transform_x(extract_x)
        return (extract_x, extract_y.ravel())

    def _infer_size(self, features, targets):
        """Function to determine the size of inputs and outputs

        Parameters
        ----------
        dataset : _type_
            _description_
        """
        self.data_size = len(features)
        self.input_size = features.shape[1]
        self.output_size = targets.shape[1]

    def _post_process(self, data):
        if self.scaler is not None:
            processed = self.scaler.inverse_transform(data)
            return processed
        else:
            return data

    def get_datasize(self, data):
        if self.data_size is not None:
            return self.data_size
        return len(data.features[0])
    