import pathlib

import pytest
import torch
from PIL import Image, ImageChops

from basnet.image_processing_basnet import BASNetImageProcessor
from basnet.modeling_basnet import BASNetModel, convert_from_checkpoint


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


def test_load_pretrained_model(repo_id: str, checkpoint_filename: str):
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
    )
    assert isinstance(model, BASNetModel)
    assert model.training is False


@pytest.mark.parametrize(
    argnames="input_filename, expected_filename",
    argvalues=(
        ("0003.jpg", "0003.png"),
        ("0005.jpg", "0005.png"),
        ("0010.jpg", "0010.png"),
        ("0012.jpg", "0012.png"),
    ),
)
def test_basnet(
    checkpoint_path: pathlib.Path,
    test_input_image_dir: pathlib.Path,
    test_output_image_dir: pathlib.Path,
    input_filename: str,
    expected_filename: str,
    device: torch.device,
):
    model = convert_from_checkpoint(checkpoint_path)
    model = model.to(device)  # type: ignore
    processor = BASNetImageProcessor()

    input_filepath = test_input_image_dir / input_filename
    output_filepath = test_output_image_dir / expected_filename

    image = Image.open(input_filepath)
    width, height = image.size

    inputs = processor(images=image)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output, *_ = model(**inputs)
    assert list(output.shape) == [1, 1, 256, 256]

    image = processor.postprocess(output, width=width, height=height)

    expected_image = Image.open(output_filepath)

    diff = ImageChops.difference(image, expected_image)
    assert diff.getbbox() is None
