from transformers.configuration_utils import PretrainedConfig


class BASNetConfig(PretrainedConfig):
    model_type = "basnet"

    def __init__(
        self,
        n_channels: int = 3,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.kernel_size = kernel_size
