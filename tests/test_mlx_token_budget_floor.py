def test_latent_bridge_never_reduces_one_token_budget_to_zero(monkeypatch):
    from core.brain import latent_bridge

    substrate = {
        "vitality": 0.0,
        "phi": 0.0,
        "free_energy": 0.5,
        "acetylcholine": 0.5,
        "serotonin": 0.5,
        "norepinephrine": 0.5,
        "cortisol": 1.0,
        "frustration": 0.0,
        "curiosity": 0.5,
        "valence": 0.0,
        "arousal": 0.5,
    }
    monkeypatch.setattr(latent_bridge, "_read_substrate", lambda: dict(substrate))

    params = latent_bridge.compute_inference_params(base_max_tokens=1)

    assert params.max_tokens == 1


def test_mlx_client_token_budget_floor_blocks_zero_token_requests():
    from core.brain.llm.mlx_client import _bounded_max_tokens

    assert _bounded_max_tokens(1, 0, 4096) == 1
    assert _bounded_max_tokens(0, 0, 4096) == 1
    assert _bounded_max_tokens(256, 128, 4096) == 128
