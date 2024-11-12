import pytest
import torch
from torch import nn
from model.minkunet_encoder import MinkUNetEncoder

def test_initialization():
    # Test default initialization
    encoder = MinkUNetEncoder()
    assert encoder.embed_dim == 384
    assert encoder.freeze_encoder_weights is False
    assert encoder.projection_type == 'linear'

    # Test invalid projection type
    with pytest.raises(ValueError):
        encoder = MinkUNetEncoder(projection_type="unsupported_type")

def test_set_projection_type():
    encoder = MinkUNetEncoder()

    # Test setting projection to linear
    encoder.set_projection_type()
    assert isinstance(encoder.projection, nn.Linear)

    # Test setting projection to mlp
    encoder.projection_type = "mlp"
    encoder.set_projection_type()
    assert isinstance(encoder.projection, nn.Sequential)

    # Test invalid projection type
    encoder.projection_type = "invalid"
    with pytest.raises(ValueError):
        encoder.set_projection_type()

def test_forward_pass():
    encoder = MinkUNetEncoder().cuda()

    # Create a batch of dummy point clouds
    point_clouds = [torch.rand(1000, 4).cuda() for _ in range(8)]  # batch size of 8
    embeddings = encoder(point_clouds)

    # Ensure output has the correct shape
    assert embeddings.shape == (8, encoder.embed_dim)

def test_freeze_encoder_weights():
    encoder = MinkUNetEncoder(freeze_encoder_weights=True)

    # Ensure encoder parameters are frozen but projection is trainable
    for name, param in encoder.named_parameters():
        if 'projection' in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Test the case when freeze_encoder_weights=False
    encoder = MinkUNetEncoder(freeze_encoder_weights=False)
    for param in encoder.parameters():
        assert param.requires_grad is True
