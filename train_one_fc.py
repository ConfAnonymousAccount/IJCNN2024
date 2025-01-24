import os
import argparse
import shutil
import sys
import json
import pathlib
import numpy as np

#os.chdir("/home/ubuntu/SYSTEMX/milad/GNN/")
from package.data.lifeline_dataset import LifeLineDataSet
from package.data.data_manager import DataManager

from package.utils import NpEncoder
from package.model.fully_connected import FullyConnected
from package.model.torch_models import TorchModels
from package.model.utils import compute_metrics

def create_parser():
    this_parser = argparse.ArgumentParser()
    this_parser.add_argument("--epochs", type=int, default=200,
                             help="Number of training epoch")
    this_parser.add_argument("--name", type=str, default="test",
                             help="Name of the model")
    #this_parser.add_argument("--nb_model", type=int, default=1,
    #                         help="number of model to train")
    this_parser.add_argument("--learning_rate", type=float, default=3e-4,
                             help="Learning rate")
    this_parser.add_argument("--activation", type=str, default="relu",
                             help="activation function [relu, tanh, sigmoid]")
    # this_parser.add_argument("--loss", type=str, default="mse",
    #                          help="loss function used for training [mse]")
    this_parser.add_argument("--train_batch_size", type=int, default=128,
                             help="Minibatch size")
    this_parser.add_argument("--device", type=str, default="cpu",
                             help="Device to be used for the training of models")

    # NN shape
    # this_parser.add_argument("--encoder_size", type=int, default=20,
    #                          help="Number of units (per layer) for the 'encoder' layer")
    # this_parser.add_argument("--nb_layer_enc", type=int, default=3,
    #                          help="Number of layers for the 'main' layer")
    this_parser.add_argument("--hidden_size", type=int, default=300,
                             help="Number of units (per layer) for the 'GCN' layer")
    this_parser.add_argument("--nb_layer_hidden", type=int, default=3,
                             help="Number of layers for the 'GCN' layer")
    # this_parser.add_argument("--decoder_size", type=int, default=40,
    #                          help="Number of units (per layer) for the 'decoder' layer")
    # this_parser.add_argument("--nb_layer_dec", type=int, default=2,
    #                          help="Number of layers for the 'decoder' layer")

    return this_parser

def read_data():
    data_path = pathlib.Path().absolute().parent.parent
    dataset = LifeLineDataSet(data_path=data_path / "data" / "extract" / "merged", 
                            dataset_name="merged_vaccin_only_1_full.csv",
                            target_variable="long_covid_intensity")
    dataset.get_encoded_data()

    data_manager = DataManager()
    #data_manager.split_data_train_test(dataset)
    data_manager.split_data_train_val_test(dataset, train_size=0.7, val_test_prop=0.8)

    return data_manager

def main(config_path,
         config_name,
         name: str = "test",
         #nb_model: int = 1,
         hidden_size=50,
         nb_layer_hidden=5,
         activation="relu",
         learning_rate=3e-4,
         train_batch_size=128,
         epochs=3,
         device="cpu",
         path_save: str=None,
         command: str=None
        ):
    
    if path_save is None:
        path_save = os.path.join("fine_tuning_fc")
        if not os.path.exists(path_save):
            os.mkdir(path_save)

    data_manager = read_data()
    train_features = data_manager.train_dataset.get_features()
    val_features = data_manager.val_dataset.get_features()
    test_features = data_manager.test_dataset.get_features()

    train_targets = data_manager.train_dataset.get_targets()
    val_targets = data_manager.val_dataset.get_targets()
    test_targets = data_manager.test_dataset.get_targets()

    hidden_sizes = [int(hidden_size) for _ in range(nb_layer_hidden)]
    fc_model = TorchModels(FullyConnected,
                           config_path=config_path,
                           config_name=config_name,
                           name=name,
                           layers=hidden_sizes,
                           activation=activation,
                           epochs=epochs)
    
    print("epochs: ", epochs)
    print("device: ", device)

    #optimizer = get_optimizer(model, learning_rate)
    fc_model.train(train_dataset=data_manager.train_dataset,
                   val_dataset=data_manager.val_dataset,
                   lr=learning_rate
                   )

    predictions = fc_model.predict(data_manager.val_dataset, scale_only_features=False) 
    metric_dict = compute_metrics(data_manager.val_dataset.targets, predictions, index=-1)
    #pred, true, metric_dict, loss = evaluate_model(model, test_loader, test_targets, metrics=metrics)

    metric_dict["neural net"] = {
        "hidden_sizes": [int(el) for el in hidden_sizes],
        "activation": str(activation),
        "lr": float(learning_rate),
        "batch_size": train_batch_size,
        "epochs": epochs,
        "command": command
    }
    metric_dict["train_losses"] = fc_model.train_losses
    metric_dict["val_losses"] = fc_model.val_losses

    #save_json = os.path.join(path_save, name + str(nb_model))
    #if not os.path.exists(save_json):
    #    os.mkdir(save_json)
    #save_json = os.path.join(path_save, name + str(nb_model) + "_metrics.json")
    save_json = os.path.join(path_save, name)
    if not os.path.exists(save_json):
        os.mkdir(save_json)
    save_json = os.path.join(path_save, name, "metrics.json")
    with open(save_json, encoding="utf-8", mode="w") as f:
        json.dump(fp=f, obj=metric_dict, indent=4, sort_keys=True, cls=NpEncoder)

    return fc_model.val_losses[-1]

if __name__ == "__main__":
    CONFIG_PATH = pathlib.Path().resolve() / "configurations" / "models" / "torch_fc.ini"
    CONFIG_NAME = "DEFAULT"
    
    import sys
    parser = create_parser()
    args = parser.parse_args()
    # command used to run the script
    command = ' '.join([sys.executable] + sys.argv)

    main(config_path=CONFIG_PATH,
         config_name=CONFIG_NAME,
         name=str(args.name),
         #nb_model=int(args.nb_model),
         hidden_size=int(args.hidden_size),
         nb_layer_hidden=int(args.nb_layer_hidden),
         activation=str(args.activation),
         learning_rate=float(args.learning_rate),
         train_batch_size=int(args.train_batch_size),
         epochs=int(args.epochs),
         device=str(args.device),
         command=command
         )
