import os
import sys

from error_checks import validate_request_data
from rdi_liveness_detection import initialize_model, get_release_version
from utils import (
    read_config,
    create_app,
    register_swagger,
    prepare_request_data,
    get_check_result,
)
from flask import jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "static", "docs.yml")
CONFIG_DIR = os.path.join(BASE_DIR, "config.json")
CONFIG, VALID = read_config(BASE_DIR, CONFIG_DIR)

if not VALID:
    print(CONFIG["error"])
    sys.exit(1)

SERVER_PORT = CONFIG["server_port"]
VERSION = CONFIG["version"]
WORKING_DIR = CONFIG["working_dir"]
MODEL_PATH = CONFIG["model_path"]

app = create_app()
app = register_swagger(app, DOCS_PATH)
# initialize_model(MODEL_PATH)


@app.route("/alive", methods=["GET"])
def alive_view():
    """
    This function is attached to the /alive route.

    Returns
    ========
        flask.wrappers.Response
            A json response contains a message key:
                {"message": "alive"}

    """
    return jsonify({"message": "alive"})


@app.route("/get_release_version", methods=["GET"])
def get_release_version_view():
    """
    This function is /get_release_version to the alive route.

    Returns
    ========
        flask.wrappers.Response
            A json response contains a version key:
                {"version": "v1.1.1.1"}

    """
    cpp_engine_version = get_release_version().replace("e","").replace("m","")
    return jsonify({"version": f"{VERSION}-{cpp_engine_version}"})


@app.route("/get_remaining_units", methods=["GET"])
def get_remaining_units_view():
    """This function is used to check the remaining units
    Returns:
    res Json : Remaining units code 200
    """
    try:
        from rdi_liveness_detection import get_remaining_units

        return jsonify({"remaining_units": get_remaining_units()})
    except ImportError:
        from exceptions_defination import Validation

        raise Validation(4013)


@app.route("/liveness_check", methods=["POST"])
def liveness_check_view():
    validate_request_data(request.json)
    prepared_data = prepare_request_data(request.json)
    return get_check_result(WORKING_DIR, MODEL_PATH, prepared_data)


if __name__ == "__main__":
    app.run(
        debug=int(os.getenv("FLASK_DEBUG", 0)) == 1,
        host="0.0.0.0",
        port=SERVER_PORT,
        threaded=True,
    )
