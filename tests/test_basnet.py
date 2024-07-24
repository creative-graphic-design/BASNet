import pathlib
from typing import Optional

import pytest
import torch
from PIL import Image, ImageChops

from basnet import BASNetImageProcessor, convert_from_checkpoint


@pytest.mark.parametrize(
    argnames="input_filename, expected_filename",
    argvalues=(
        ("0003.jpg", "0003.png"),
        ("0005.jpg", "0005.png"),
        ("0010.jpg", "0010.png"),
        ("0012.jpg", "0012.png"),
    ),
)
@pytest.mark.parametrize(
    argnames="return_dict",
    argvalues=(True, False, None),
)
def test_basnet(
    repo_id: str,
    checkpoint_filename: str,
    test_input_image_dir: pathlib.Path,
    test_output_image_dir: pathlib.Path,
    input_filename: str,
    expected_filename: str,
    device: torch.device,
    return_dict: Optional[bool],
):
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
    )
    model = model.to(device)  # type: ignore
    processor = BASNetImageProcessor()

    input_filepath = test_input_image_dir / input_filename
    output_filepath = test_output_image_dir / expected_filename

    image = Image.open(input_filepath)
    width, height = image.size

    inputs = processor(images=image)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, return_dict=return_dict)

    prediction = outputs.activated.dout if return_dict else outputs[0][0]

    assert list(prediction.shape) == [1, 1, 256, 256]

    image = processor.postprocess(prediction, width=width, height=height)

    expected_image = Image.open(output_filepath)

    diff = ImageChops.difference(image, expected_image)
    assert diff.getbbox() is None
