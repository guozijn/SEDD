import json

from sedd_mini.server import create_app


def test_models_endpoint_exposes_registry_entries(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "default_model_id": "arc_lora_sft",
                "models": [
                    {"id": "base", "label": "SEDD small base", "backend": "official"},
                    {"id": "arc_lora_sft", "label": "ARC LoRA SFT", "backend": "official"},
                    {"id": "arc_dcolt_rl", "label": "ARC DCoLT RL", "backend": "official"},
                ],
            }
        ),
        encoding="utf-8",
    )

    app = create_app("", model_registry=str(registry_path))
    models_route = next(route for route in app.routes if getattr(route, "path", None) == "/models")
    payload = models_route.endpoint()

    assert payload["default_model_id"] == "arc_lora_sft"
    assert [model["id"] for model in payload["models"]] == ["base", "arc_lora_sft", "arc_dcolt_rl"]


def test_models_endpoint_auto_discovers_arc_registry(tmp_path, monkeypatch):
    registry_dir = tmp_path / "runs" / "arc_models"
    registry_dir.mkdir(parents=True)
    registry_path = registry_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "default_model_id": "arc_lora_sft",
                "models": [
                    {"id": "base", "label": "SEDD-small base", "backend": "official"},
                    {"id": "arc_lora_sft", "label": "ARC LoRA SFT", "backend": "official"},
                    {"id": "arc_dcolt_rl", "label": "ARC DCoLT RL", "backend": "official"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    app = create_app("")
    models_route = next(route for route in app.routes if getattr(route, "path", None) == "/models")
    payload = models_route.endpoint()

    assert payload["default_model_id"] == "arc_lora_sft"
    assert [model["id"] for model in payload["models"]] == ["base", "arc_lora_sft", "arc_dcolt_rl"]
