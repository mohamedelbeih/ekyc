from jsonschema import Draft7Validator


def validate_request_data(input_json: dict) -> None:
    """
    This function validates the json input, raises jsonschema.exceptions.ValidationError if not valid.

    Parameters
    ===========
        input_json: dict
            The input json sent in the request

    Returns
    ========
        None
            always

    Raises
    =======
        jsonschema.exceptions.ValidationError
            If input not valid
    """
    schema = {
        "type": "object",
        "properties": {
            "video": {"type": "string"},
            "actions": {"type": "array"},
            "actions_times": {"type": "array"},
        },
        "required": ["video", "actions"],
    }
    Draft7Validator(schema).validate(input_json)
