from core.cognitive_integration_layer import CognitiveIntegrationLayer


def test_sync_setup_creates_base_directory_and_status(tmp_path):
    base_dir = tmp_path / "cognition"
    layer = CognitiveIntegrationLayer(base_data_dir=str(base_dir))

    assert layer.setup() is True
    assert base_dir.exists()
    assert layer.get_status()["setup_complete"] is True
