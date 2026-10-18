from fastapi import APIRouter

from schemalens_api.deps import SettingsDep
from schemalens_api.models import PresetField, PresetInfo, PresetList
from schemalens_core.presets import load_presets

router = APIRouter(tags=["extraction"])


@router.get(
    "/api/v1/presets",
    operation_id="list_presets",
    summary="List schema presets",
)
def list_presets(settings: SettingsDep) -> PresetList:
    """Ready-made schemas. Use a preset by name in `preset`, optionally extended by `schema`."""
    presets = []
    for preset in load_presets(settings).values():
        required = set(preset.json_schema.get("required", []))
        fields = [
            PresetField(
                name=name,
                type=str(node.get("type", "string")),
                description=node.get("description"),
                derived=node.get("x-schemalens-derived") is True,
                semantic_type=node.get("x-schemalens-type"),
                required=name in required,
            )
            for name, node in preset.json_schema["properties"].items()
        ]
        presets.append(
            PresetInfo(
                name=preset.name,
                version=preset.version,
                title=preset.title,
                description=preset.description,
                fields=fields,
            )
        )
    return PresetList(presets=presets)
