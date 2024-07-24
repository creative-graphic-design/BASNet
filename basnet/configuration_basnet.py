from transformers.configuration_utils import PretrainedConfig


class BASNetConfig(PretrainedConfig):
    model_type = "basnet"

    def __init__(
        self,
        resnet_model: str = "microsoft/resnet-34",
        n_channels: int = 3,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.resnet_model = resnet_model
        self.n_channels = n_channels

        self.kernel_size = 3
