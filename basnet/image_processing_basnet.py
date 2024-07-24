from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from torchvision import transforms
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput


class RescaleT(object):
    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample) -> Dict[str, np.ndarray]:
        image, label = sample["image"], sample["label"]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        # img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        img = (
            cv2.resize(
                image,
                (self.output_size, self.output_size),
                interpolation=cv2.INTER_AREA,
            )
            / 255.0
        )
        # lbl = transform.resize(label, (self.output_size, self.output_size),
        #                        mode='constant',
        #                        order=0,
        #                        preserve_range=True)
        lbl = cv2.resize(
            label, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST
        )
        lbl = np.expand_dims(lbl, axis=-1)
        lbl = np.clip(lbl, np.min(label), np.max(label))

        return {"image": img, "label": lbl}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag: int = 0) -> None:
        self.flag = flag

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            # tmpImgtl = color.rgb2lab(tmpImgt)
            tmpImgtl = cv2.cvtColor(tmpImgt, cv2.COLOR_RGB2LAB)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2])
            )
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0])
            )
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1])
            )
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2])
            )

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(
                tmpImg[:, :, 3]
            )
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(
                tmpImg[:, :, 4]
            )
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(
                tmpImg[:, :, 5]
            )

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            # tmpImg = color.rgb2lab(tmpImg)
            print("tmpImg:", tmpImg.min(), tmpImg.max())
            exit()
            tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_RGB2LAB)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0])
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1])
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2])
            )

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(
                tmpImg[:, :, 0]
            )
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(
                tmpImg[:, :, 1]
            )
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(
                tmpImg[:, :, 2]
            )

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {"image": torch.from_numpy(tmpImg), "label": torch.from_numpy(tmpLbl)}


def apply_transform(
    data: Dict[str, np.ndarray], rescale_size: int, to_tensor_lab_flag: int
) -> Dict[str, torch.Tensor]:
    transform = transforms.Compose(
        [RescaleT(output_size=rescale_size), ToTensorLab(flag=to_tensor_lab_flag)]
    )
    return transform(data)  # type: ignore


class BASNetImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self, rescale_size: int = 256, to_tensor_lab_flag: int = 0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.rescale_size = rescale_size
        self.to_tensor_lab_flag = to_tensor_lab_flag

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        if not isinstance(images, PilImage):
            raise ValueError(f"Expected PIL.Image, got {type(images)}")

        image_pil = images
        image_npy = np.array(image_pil, dtype=np.uint8)
        width, height = image_pil.size
        label_npy = np.zeros((height, width), dtype=np.uint8)

        assert image_npy.shape[-1] == 3
        output = apply_transform(
            {"image": image_npy, "label": label_npy},
            rescale_size=self.rescale_size,
            to_tensor_lab_flag=self.to_tensor_lab_flag,
        )
        image = output["image"]

        assert isinstance(image, torch.Tensor)

        return BatchFeature(
            data={"pixel_values": image.float().unsqueeze(dim=0)}, tensor_type="pt"
        )

    def postprocess(
        self, prediction: torch.Tensor, width: int, height: int
    ) -> PilImage:
        def _norm_prediction(d: torch.Tensor) -> torch.Tensor:
            ma, mi = torch.max(d), torch.min(d)

            # division while avoiding zero division
            dn = (d - mi) / ((ma - mi) + torch.finfo(torch.float32).eps)
            return dn

        prediction = _norm_prediction(prediction)
        prediction = prediction.squeeze()
        prediction = prediction * 255 + 0.5
        prediction = prediction.clamp(0, 255)

        prediction_np = prediction.cpu().numpy()
        image = Image.fromarray(prediction_np).convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        return image
