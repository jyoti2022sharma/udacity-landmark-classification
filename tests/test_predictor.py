######################################################################################
#                                     TESTS
######################################################################################
import pytest
import torch
from src.data import get_data_loaders
from src.model import MyModel
from src.helpers import compute_mean_and_std
from src.predictor import Predictor


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"