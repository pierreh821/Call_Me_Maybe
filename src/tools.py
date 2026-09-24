from pydantic import BaseModel, field_validator


class TypeHandler:
    types = {
        "number": float,
        "string": str,
        "integer": int
        }


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, type | None]

    @field_validator("parameters", mode='before')
    def _format_parameters(cls, raw_param: dict[str, dict[str, str]]
                           ) -> dict[str, type | None]:
        if not isinstance(raw_param, dict):
            raise ValueError("Parameters must be a dictionary")

        cln_param: dict[str, type | None] = {}
        for name, details in raw_param.items():
            if not isinstance(details, dict) or "type" not in details:
                raise ValueError(f"Invalid parameter structure for '{name}'")

            type_str = details.get("type") or ""
            type_cln = TypeHandler.types.get(type_str)
            if type_cln is None:
                raise ValueError(f"Unsupported or missing type: '{type_str}'")

            cln_param[name] = type_cln

        return cln_param
