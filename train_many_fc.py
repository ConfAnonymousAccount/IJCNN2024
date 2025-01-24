import datetime
import nevergrad as ng
import sys
import subprocess
import json
import os
import argparse

BUDGET = 100


def create_parser():
    this_parser = argparse.ArgumentParser()
    this_parser.add_argument("--budget", type=int, default=BUDGET,
                             help=f"Number of neural network that will be trained "
                                  f"default={BUDGET}")
    return this_parser


def train_model_seq(**kwargs):
    """train one model on one GPU"""
    name = f"{datetime.datetime.now():%Y%m%d_%H%M}"
    subprocess.run([sys.executable, "train_one_fc.py",
                    "--name", name,
                    "--learning_rate", f'{kwargs["learning_rate"]}',
                    "--activation", f'{kwargs["activation"]}',
                    "--hidden_size", f'{kwargs["hidden_size"]}',
                    "--nb_layer_hidden", f'{kwargs["nb_layer_hidden"]}',
                    # "--encoder_size", f'{kwargs["encoder_size"]}',
                    # "--nb_layer_enc", f'{kwargs["nb_layer_enc"]}',
                    # "--decoder_size", f'{kwargs["decoder_size"]}',
                    # "--nb_layer_dec", f'{kwargs["nb_layer_dec"]}',
                    "--epochs", "100",
                    ])
    with open(os.path.join("fine_tuning_fc", name, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    val_loss = metrics["val_losses"][-1]
    return val_loss


def main(budget=BUDGET):
    # this code is inspired from the nevergrad documentation available at:
    # https://github.com/facebookresearch/nevergrad
    # https://facebookresearch.github.io/nevergrad/machinelearning.html#ask-and-tell-version

    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        learning_rate=ng.p.Log(lower=1e-5, upper=1e-2),
        # encoder_size=ng.p.Scalar(lower=5, upper=20).set_integer_casting(),
        # nb_layer_enc=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
        hidden_size=ng.p.Scalar(lower=50, upper=300).set_integer_casting(),
        nb_layer_hidden=ng.p.Scalar(lower=1, upper=5).set_integer_casting(),
        # decoder_size=ng.p.Scalar(lower=20, upper=150).set_integer_casting(),
        # nb_layer_dec=ng.p.Scalar(lower=0, upper=3).set_integer_casting(),
        activation=ng.p.Choice(["relu", "sigmoid", "tanh"])
    )

    optim = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    for _ in range(budget):
        x1 = optim.ask()
        res = train_model_seq(**x1.kwargs)
        optim.tell(x1, res)
    recommendation = optim.recommend()
    train_model_seq(**recommendation.kwargs)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(budget=int(args.budget))
