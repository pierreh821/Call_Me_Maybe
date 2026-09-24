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
        cln_param = {}
        for item in raw_param.items():
            name = item[0]
            type_str = item[1].get("type")
            if type_str is not None:
                type_cln = TypeHandler.types.get(type_str)
            else:
                raise SyntaxError("Type not specified in function returns")
            cln_param[name] = type_cln

        return cln_param

    @staticmethod
    def is_valid(raw: dict) -> bool:
        if not (
                "name" in raw.keys()
                and "description" in raw.keys()
                and "parameters" in raw.keys()):
            return False

        if not isinstance(raw.get("parameters"), dict):
            return False

        if len(raw.get("parameters", {})) > 0:
            for opt in raw.get("parameters", {}).items():
                if not isinstance(opt[1], dict):
                    return False

        return True
