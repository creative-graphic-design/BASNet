from basnet import BASNetModel, convert_from_checkpoint


def test_load_pretrained_model(repo_id: str, checkpoint_filename: str):
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
    )
    assert isinstance(model, BASNetModel)
    assert model.training is False
