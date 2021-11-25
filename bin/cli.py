from datetime import datetime
from pathlib import Path
import unittest

import typer
import torch
import git

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.infer import infer as run_infer
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize, flip

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    run_path: Path = typer.Option(
        ..., "-p", "--path", help="Path to run from which you want to infer."),
    label: int = typer.Option(
        6, "-l", "--label", help="Label of image to show to the model"),
    # Switch on and off transforms via typer 
    norm_bool : bool = typer.Option(
        True, "-n", "--normalise", help = "Activate Normalise transforms"),
    
    flip_bool : bool = typer.Option(
        True, "-f", "--flip", help = "Activate Flip transforms")
    ):
    
    """Run the inference code"""
    params = load_params(run_path)
    model = torch.load(run_path / "model.pt")
    image = _select_test_image(label, norm_bool, flip_bool)
    run_infer(params, model, image, label)


def _select_test_image(label, norm_bool, flip_bool):
    # DONE
    
    if norm_bool and flip_bool:
        ts = [normalize, flip]
    elif norm_bool and not flip_bool: 
        ts = [normalize]
    elif not norm_bool and flip_bool: 
        ts = [flip]
    else: 
        raise TransformDefinitionError("Need to define a Transform")
    dataloader = test_dataloader(1, transforms(*ts))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images

class TransformDefinitionError(Exception):
    pass

class Test(unittest.TestCase):
    
    # Check function returns what we want it to return with fixed label 
    def test_fun_normalize(self):
        label = 6
        ts = [flip]
        dataloader_n = test_dataloader(1, transforms(*ts))
        images, labels = next(iter(dataloader_n))
        while labels[0].item() != label:
            images, labels = next(iter(dataloader_n))
        return images

    def test_select_test_image(self): 
        assert torch.all(torch.eq(_select_test_image(6, False, True), self.test_fun_normalize()))
        
if __name__ == "__main__":
    unittest.main()    
