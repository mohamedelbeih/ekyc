import json
import os
import sys

from flask import Blueprint, jsonify
from jsonschema.exceptions import ValidationError

from exceptions_defination import (
    Validation,
    InternalServerError,
    SomethingWrongHappened,
)


def read_errors(path: str) -> (dict, bool):
    """
    This function reads the file containing the error messages.

    Parameters
    ===========
        path: str
            The input file containing the error messages

    Returns
    ========
        (dict, bool)
            if bool is False, dict will contain an error key containing the error message like:
                {'error': f"Error file not found,...."}
            if bool is True, dic will contain a dict contain the error messages in the same form as the passed json file.

    """
    if not os.path.isfile(path):
        return {"error": f"Error file not found, should be at {path}"}, False
    try:
        errors = json.load(open(path, encoding="utf-8"))
        return errors, True
    except ValueError:
        return {"error": f"Couldn't parse errors while reading file {path}"}, False


blueprint = Blueprint("error_handlers", __name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERRORS, VALID = read_errors(os.path.join(BASE_DIR, "errors.json"))
if not VALID:
    print(ERRORS["error"])
    sys.exit(1)


def construct_error_response(internal_code: int) -> dict:
    """
    This function constructs an error response

    Parameters
    ===========
        internal_code: string
            error code

    Returns
    ========
        a dict with the following fields
        {
            "error": error message,
            "description": error description
        }
    """
    return {"error_code": internal_code, **ERRORS[f"{internal_code}"]}


@blueprint.app_errorhandler(404)
def non_existent_target_error_handler(_):
    """
    Catches all 404 errors raised by flask.

    Parameters
    ===========
        _
            The error object passes automatically by flask, we assign it to _ because we don't need it.
    Returns
    ========
        flask.wrappers.Response, 404
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(4015)
    return jsonify(response), 404


@blueprint.app_errorhandler(405)
def not_allowed_method_error_handler(_):
    """
    Catches all 405 (wrong method) errors raised by flask.

    Parameters
    ===========
        _
            The error object passes automatically by flask, we assign it to _ because we don't need it.
    Returns
    ========
        flask.wrappers.Response, 405
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(4010)
    return jsonify(response), 405


@blueprint.app_errorhandler(Validation)
def validation_error_handler(error):
    """
    Catches our custom validation error raised by us in any place in the code.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 400
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 400


@blueprint.app_errorhandler(ValidationError)
def inp_json_error_handler(error):
    """
    Catches jsonschema.exceptions.ValidationError raised by jsonschema when validating request.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 400
            The response contains a nice message, and the status code is for the client.
    """
    description = error.message
    error_code = 4012
    if "is not of type" in description:
        error_code = 4011
    elif "a required property" in description:
        if "video" in description:
            error_code = 4008
        elif "actions" in description:
            error_code = 4014
    response = construct_error_response(error_code)
    return jsonify(response), 400


@blueprint.app_errorhandler(InternalServerError)
def internal_server_error_handler(error):
    """
    Catches our custom InternalServerError raised by us in any part of the code.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 500
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 500


@blueprint.app_errorhandler(SomethingWrongHappened)
def something_wrong_happened_handler(error):
    """
    Catches our custom SomethingWrongHappened raised by us in any part of the code.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 449
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 449


@blueprint.app_errorhandler(500)
def all_other_500_errors(_):
    """
    Catches all internal server errors raised by flask.

    Parameters
    ===========
        _
            The error object passes automatically by flask, we assign it to _ because we don't need it.
    Returns
    ========
        flask.wrappers.Response, 400
            The response contains a nice message, and the status code is for the client.
    """
    response = construct_error_response(4009)
    return jsonify(response), 500
