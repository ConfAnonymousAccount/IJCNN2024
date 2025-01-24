import os
import pathlib
from typing import Union

import numpy as np
import torch
from torch import optim
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from package.data.scaler import StandardScaler

LOSSES = {"mse": nn.MSELoss,
          "mae": nn.L1Loss}

OPTIMIZERS = {"adam": optim.Adam, "sgd": optim.SGD}

class TorchModels(object):
    def __init__(self,
                 model: nn.Module,
                 config_path: Union[pathlib.Path, str],
                 config_name: str="DEFAULT",
                 name: Union[str, None]=None,
                 scaler: Union[StandardScaler, None]=None,
                 indices_to_scale: Union[list, None]=None,
                 seed: int=42,
                 **kwargs):
        super().__init__()
        
        self.trained = False
        self.params = kwargs
        self.model = model
        self._model = self.model(name=name, 
                                 config_path=config_path, 
                                 config_name=config_name, 
                                 scaler=scaler, 
                                 indices_to_scale=indices_to_scale, 
                                 **kwargs)
        self.params.update(self._model.params)
        config_name = self._model.config.section_name
        self.name = name + '_' + config_name
        self.device = torch.device(self.params["device"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self._observations = dict()
        self._predictions = dict()

        # history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}

    def build_model(self):
        """Build the model structure
        """
        self._model.build_model()

    def train(self,
              train_dataset,
              val_dataset = None,
              save_path: Union[None, str] = None,
              scale_only_features: bool=True,
              **kwargs):
        
        self.params.update(kwargs)
        train_loader = self._model.process_dataset(train_dataset, training=True, scale_only_features=scale_only_features)
        if val_dataset is not None:
            val_loader = self._model.process_dataset(val_dataset, training=False, scale_only_features=scale_only_features)

        # build the model
        self.build_model()
        self._model.to(self.params["device"])
        self._model.float()
        
        if kwargs.get("lr") is not None:
            self.params["optimizer"]["params"]["lr"] = kwargs.get("lr")
        optimizer = self._get_optimizer(optimizer=OPTIMIZERS[self.params["optimizer"]["name"]],
                                        **self.params["optimizer"]["params"])
        loss_func = self._get_loss_func()
        for metric_ in self.params["metrics"]:
            self.train_metrics[metric_] = list()
            if val_dataset is not None:
                self.val_metrics[metric_] = list()

        #self.logger.info("Training of {%s} started", self.name)
        #losses, elapsed_time = train_model(self.model, data_loaders=data)
        for epoch in range(1, self.params["epochs"]+1):
            train_loss_epoch, train_metrics_epoch = self._train_one_epoch(epoch, train_loader, optimizer, loss_func)
            self.train_losses.append(train_loss_epoch)
            for nm_, arr_ in self.train_metrics.items():
                arr_.append(train_metrics_epoch[nm_])

            if val_dataset is not None:
                val_loss_epoch, val_metrics_epoch = self._validate(val_loader, loss_func)
                self.val_losses.append(val_loss_epoch)
                for nm_, arr_ in self.val_metrics.items():
                    arr_.append(val_metrics_epoch[nm_])

            # check point
            if self.params["save_freq"] and (save_path is not None):
                if epoch % self.params["ckpt_freq"] == 0:
                    self.save(save_path, epoch)

        self.trained = True
        # save the final model
        if save_path:
            self.save(save_path)
    
    def _train_one_epoch(self, epoch: int, train_loader, optimizer, loss_func):
        """
        train the model at a epoch
        """
        data_size = len(train_loader.dataset)
        self._model.train()
        torch.set_grad_enabled(True)
        device = self.params["device"]
        total_loss = 0
        metric_dict = dict()


        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        for batch in train_loader:
            #data, edge_index, target, edge_attr = snapshot.x, snapshot.edge_index, snapshot.y, snapshot.edge_attr
            #loss_func = self._get_loss_func()
            #data, target = data.to(self.params["device"]), target.to(self.params["device"])
            optimizer.zero_grad()
            # h_0 = self.model.init_hidden(data.size(0))
            # prediction, _ = self.model(data, h_0)
            #prediction = self._model(data, edge_index, edge_attr)
            prediction, target = self._model._do_forward(batch, device=device)
            loss = loss_func(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(target)
            for metric in self.params["metrics"]:
                metric_func = LOSSES[metric](reduction="mean")
                metric_value = metric_func(prediction, target)
                metric_value = metric_value.item()*len(target)
                metric_dict[metric] += metric_value
                
        mean_loss = total_loss/data_size#len(train_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= data_size#len(train_loader.dataset)
        print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])
        return mean_loss, metric_dict

    def _validate(self, val_loader, loss_func, **kwargs):
        """function used for validation of the model

        It is separated from evaluate function, because it should be called at each epoch during training

        Parameters
        ----------
        val_loader : DataLoader
            _description_

        Returns
        -------
        set
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.params.update(kwargs)
        self._model.eval()
        device = self.params["device"]
        total_loss = 0
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        with torch.no_grad():
            for batch in val_loader:
                prediction, target = self._model._do_forward(batch, device=device)
                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(target)

                for metric in self.params["metrics"]:
                    metric_func = LOSSES[metric](reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(target)
                    metric_dict[metric] += metric_value

        mean_loss = total_loss/len(val_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(val_loader.dataset)
        print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        return mean_loss, metric_dict

    def predict(self, dataset, scale_only_features: bool=True, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        """
        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
        #self.params.update(kwargs)

        test_loader = self._model.process_dataset(dataset, training=False, scale_only_features=scale_only_features)
        data_size = self._model.get_datasize(dataset)
        # activate the evaluation mode
        self._model.eval()
        predictions = []
        observations = []
        total_loss = 0
        device = self.params["device"]
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0
        loss_func = self._get_loss_func()
        total_time = 0
        with torch.no_grad():
            for batch in test_loader:
                prediction, target = self._model._do_forward(batch, device=device)
                if device != "cpu":
                    prediction = prediction.cpu()
                    target = target.cpu()
                if not(scale_only_features):
                    prediction = self._model._post_process(prediction)
                    target = self._model._post_process(target)
                predictions.append(prediction.numpy())
                observations.append(target.numpy())

                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(target)

                for metric in self.params["metrics"]:
                    metric_func = LOSSES[metric](reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(target)
                    metric_dict[metric] += metric_value

        mean_loss = total_loss/data_size#len(test_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= data_size#len(test_loader.dataset)
        #print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
        #      [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        predictions = np.concatenate(predictions)
        observations = np.concatenate(observations)
        self._observations = observations
        self._predictions = predictions

        return predictions#mean_loss, metric_dict

    def _get_loss_func(self, loss_name: Union[None, str]=None, *args):
        if len(args) > 0:
            loss_func = None
            raise NotImplementedError("For the moment, we don't accept arguments for loss function")
        else:
            loss_func = LOSSES[self.params["loss"]["name"]](**self.params["loss"]["params"])
        return loss_func
        
    def _get_optimizer(self, optimizer: optim.Optimizer=optim.Adam, **kwargs):
        return optimizer(self._model.parameters(), **kwargs)
