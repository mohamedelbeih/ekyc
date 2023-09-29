"""
    This file 'error_handlers' used to handle the user customise errors.
    this file read the customise error from 'error.json' file after that,
    this file redirect it to the server then to the user.

"""
import json
import os
import sys

from flask import Blueprint
from flask import jsonify
from marshmallow import ValidationError

from exceptions import InternalServerError
from exceptions import SomethingWrongHappened
from exceptions import Validation


blueprint = Blueprint("error_handlers", __name__)


def read_errors(path: str) -> dict:
    """
    This function reads the file containing the error messages.

    Parameters
    ===========
        path: str
            The input file containing the error messages

    Returns
    ========
        dict
            dict containing errors

    """
    if not os.path.isfile(path):
        print(f"Error file not found, should be at {path}", flush=True)
        sys.exit(1)
    try:
        errors = json.load(open(path, encoding="utf-8"))
        return errors
    except KeyError:
        print(f"Couldn't parse errors while reading file {path}", flush=True)
        sys.exit(1)


def get_marshmallow_error_code(messages):
    """
    This function get list of marshmallow's errors,
        and redirect the first one to the user via the server.

    Inputs:
        messages:
            This is array of error massages sent from marshmallow.

    Outpus:
        first_message:
            this function send the user first error message from the list.
    """
    first_item = list(messages.values())[0]
    if isinstance(first_item, dict):
        return get_marshmallow_error_code(first_item)
    elif isinstance(first_item, list):
        return first_item[0]
    return first_item


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERRORS: dict = read_errors(os.path.join(BASE_DIR, "errors.json"))


def construct_error_response(internal_code: str) -> dict:
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
    response = construct_error_response("4404")
    return jsonify(response), 404


@blueprint.app_errorhandler(405)
def not_allowed_method_error_handler(_):
    """
    Catches all 405 (wrong method) errors raised by flask.

    Parameters
    ===========
        _
            The error object passes automatically by flask,
            we assign it to _ because we don't need it.
    Returns
    ========
        flask.wrappers.Response, 405
            The response contains a nice message
            and the status code is for the client.
    """
    response = construct_error_response("4405")
    return jsonify(response), 405


@blueprint.app_errorhandler(ValidationError)
def inp_json_error_handler(error):
    """
    This function catch marshmallow raised errors list.

    Inputs:
        error:
            This is a system raised error from marshmallow.
    Output:
        - json object:
            The response contains a nice message in json object format,
            and the status code is for the client.
    """
    error_code = get_marshmallow_error_code(error.messages)
    return jsonify(construct_error_response(error_code)), 400


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
            The response contains a nice message,
            and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 400


@blueprint.app_errorhandler(InternalServerError)
def internal_server_error_handler1(error):
    """
    Catches our custom InternalServerError raised by us in any part of the code.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 500
            The response contains a nice message,
            and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 500


@blueprint.app_errorhandler(SomethingWrongHappened)
def internal_server_error_handler(error):
    """
    Catches our custom SomethingWrongHappened raised by us in any part of the code.

    Parameters
    ===========
        error
            The error code passed when raised.
    Returns
    ========
        flask.wrappers.Response, 449
            The response contains a nice message,
            and the status code is for the client.
    """
    response = construct_error_response(error.args[0])
    return jsonify(response), 449


@blueprint.app_errorhandler(500)
def all_other_500_errors(error):
    """
    Catches all internal server errors raised by flask.

    Parameters
    ===========
        _
            The error object passes automatically by flask,
            we assign it to _ because we don't need it.
    Returns
    ========
        flask.wrappers.Response, 400
            The response contains a nice message,
            and the status code is for the client.
    """

    response = construct_error_response("4500")
    return jsonify(response), 500
