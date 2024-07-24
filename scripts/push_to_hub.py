import logging

from basnet.configuration_basnet import BASNetConfig
from basnet.image_processing_basnet import BASNetImageProcessor
from basnet.modeling_basnet import convert_from_checkpoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def push_basnet_to_hub(
    push_to_repo_name: str,
    checkpoint_filename: str,
    checkpoint_repo_name: str = "creative-graphic-design/BASNet-checkpoints",
):
    config = BASNetConfig()

    model = convert_from_checkpoint(
        repo_id=checkpoint_repo_name,
        filename=checkpoint_filename,
        config=config,
    )
    processor = BASNetImageProcessor()

    config.register_for_auto_class()
    model.register_for_auto_class()
    processor.register_for_auto_class()

    logger.info(f"Push model to the hub: {push_to_repo_name}")
    model.push_to_hub(push_to_repo_name, private=True)

    logger.info(f"Push processor to the hub: {push_to_repo_name}")
    processor.push_to_hub(push_to_repo_name, private=True)


def main():
    push_basnet_to_hub(
        checkpoint_filename="basnet.pth",
        push_to_repo_name="creative-graphic-design/BASNet",
    )
    push_basnet_to_hub(
        checkpoint_filename="gdi-basnet.pth",
        push_to_repo_name="creative-graphic-design/BASNet-SmartText",
    )


if __name__ == "__main__":
    main()
