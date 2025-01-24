import os
import pathlib
from typing import Union
import json
from copy import deepcopy

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from package.config import ConfigManager
from package.data.scaler import StandardScaler

class FullyConnected(nn.Module):

    def __init__(self,
                 config_path: Union[pathlib.Path, str],
                 config_name: Union[pathlib.Path, str],
                 name: Union[None, str]=None,
                 scaler: Union[StandardScaler, None]=None,
                 indices_to_scale: Union[list, None]=None,
                 **kwargs):
        super().__init__()
        if not os.path.exists(config_path):
            raise RuntimeError("Configuration path not found!")
        
        config_name = config_name if config_name is not None else "DEFAULT"
        self.config = ConfigManager(section_name=config_name, path=config_path)
        self.name = name if name is not None else self.config.get_option("name")
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = scaler()
        self.indices_to_scale=indices_to_scale
        self.params = self.config.get_options_dict()
        self.params.update(kwargs)

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]
        self.data_size = None

        self.input_layer = None
        self.input_dropout = None
        self.dense_layers = None
        self.dropout_layers = None
        self.output_layer = None

    def build_model(self):
        """Build the model architecture
        """
        linear_sizes = list(self.params["layers"])

        self.input_layer = nn.Linear(self.input_size, linear_sizes[0])
        self.input_dropout = nn.Dropout(p=self.params["input_dropout"])

        self.dense_layers = nn.ModuleList([nn.Linear(in_f, out_f) \
                                           for in_f, out_f in zip(linear_sizes[:-1], linear_sizes[1:])])
        
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.params["dropout"]) \
            for _ in range(len(self.dense_layers))])

        self.output_layer = nn.Linear(linear_sizes[-1], self.output_size)

    def forward(self, data):
        """FORWARD step of the model

        Parameters
        ----------
        data : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        activation = self.activation[self.params["activation"]]
        out = self.input_layer(data)
        out = activation(out)
        out = self.input_dropout(out)
        for _, (dense_, dropout) in enumerate(zip(self.dense_layers, self.dropout_layers)):
            out = dense_(out)
            out = activation(out)
            out = dropout(out)
        out = self.output_layer(out)
        return out 
    
    def process_dataset(self, dataset, training: bool, scale_only_features: bool=True):

        if training:
            batch_size = self.params["train_batch_size"]
            extract_x = deepcopy(dataset.get_features())
            extract_y = deepcopy(dataset.get_targets())
            self._infer_size(extract_x, extract_y)
            if self.scaler is not None:
                # fit the scaler
                self.scaler.fit(extract_x, extract_y, self.indices_to_scale)
                # transform the x and y variables wrt the flag
                if scale_only_features:
                    extract_x = self.scaler.transform_x(extract_x)
                else:
                    extract_x, extract_y = self.scaler.transform(extract_x, extract_y)                    
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x = deepcopy(dataset.get_features())
            extract_y = deepcopy(dataset.get_targets())
            self._infer_size(extract_x, extract_y)
            if self.scaler is not None:
                # scale the x and y variables wrt flag for other datasets than train
                if scale_only_features:
                    extract_x = self.scaler.transform_x(extract_x)
                else:
                    extract_x, extract_y =  self.scaler.transform(extract_x, extract_y)
                    

        torch_dataset = TensorDataset(torch.from_numpy(extract_x).float(), torch.from_numpy(extract_y).float())
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader
    
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
        
    def _do_forward(self, batch, device):
        data, target = batch
        data = data.to(device)
        target = target.to(device)
        predictions = self.forward(data)
        return predictions, target
