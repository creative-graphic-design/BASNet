import pathlib

import pytest
import torch


@pytest.fixture
def repo_id() -> str:
    return "creative-graphic-design/BASNet-checkpoints"


@pytest.fixture
def checkpoint_filename() -> str:
    return "basnet.pth"


@pytest.fixture
def test_fixtures_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "test_fixtures"


@pytest.fixture
def test_input_image_dir(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "images" / "inputs"


@pytest.fixture
def test_output_image_dir(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "images" / "outputs"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
