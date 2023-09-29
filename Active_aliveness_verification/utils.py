import base64
import binascii
import json
import os
import uuid
from time import time

import error_handlers
from exceptions_defination import SomethingWrongHappened
from exceptions_defination import Validation
from flasgger import Swagger
from flask import Flask
from flask import jsonify
from rdi_liveness_detection import Action_BLINKING
from rdi_liveness_detection import Action_LOOKINGCENTER
from rdi_liveness_detection import Action_LOOKINGDOWN
from rdi_liveness_detection import Action_LOOKINGLEFT
from rdi_liveness_detection import Action_LOOKINGRIGHT
from rdi_liveness_detection import Action_LOOKINGUP
from rdi_liveness_detection import Action_SMILING
from rdi_liveness_detection import ActionList
from rdi_liveness_detection import liveness_check_from_path
from rdi_liveness_detection import TimeSlotList

ACTIONS_LIST = {
    "Center": Action_LOOKINGCENTER,
    "Right": Action_LOOKINGRIGHT,
    "Left": Action_LOOKINGLEFT,
    "Up": Action_LOOKINGUP,
    "Down": Action_LOOKINGDOWN,
    "Smiling": Action_SMILING,
    "Blinking": Action_BLINKING,
}
ENGINE_ERRORS = {
    4400: 4005,
    4401: 4006,
    4402: 4007,
}


def create_app() -> Flask:
    """
    This function creates the main flask app, register the needed blueprints.

    Returns
    ========
        flask.Flask
            The main flask app instance.

    """
    app = Flask(__name__)
    app.register_blueprint(error_handlers.blueprint)
    return app


def register_swagger(app: Flask, docs_file: str) -> Flask:
    """
    This function registers swagger documentations from a yml file.

    Parameters
    ===========
        app
            The Flask app instance.
        docs_file
            The path of the yml documentations file.

    Returns
    ========'Center', 'Right', 'Left', 'Up', 'Down', 'Smiling', 'Blinking'
        flask.Flask
            The main flask app instance after registering swagger.

    """
    app.config["SWAGGER"] = {
        "uiversion": 3,
        "openapi": "3.0.2",
        "termsOfService": None,
        "favicon": "https://cdn-bcobd.nitrocdn.com/lFpPYTEQGbSBBnHvsvxLVWWYrgJcBXSt/assets/static/optimized/wp-content"
        "/uploads/2020/10/14871788ed747d9ce26080b8d3423eb5.cropped-icon-32x32.png",
    }
    swagger_config = Swagger.DEFAULT_CONFIG
    swagger_config[
        "swagger_ui_bundle_js"
    ] = "//cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.5.0/swagger-ui-bundle.min.js"
    swagger_config["ui_params"] = "plugins: [JSONBase64Reader]"
    swagger_config["swagger_ui_css"] = "//unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css"
    swagger_config[
        "swagger_ui_standalone_preset_js"
    ] = "//cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.5.0/swagger-ui-standalone-preset.min.js"
    swagger_config["jquery_js"] = "//unpkg.com/jquery@2.2.4/dist/jquery.min.js"
    swagger_config["headers"] = [("Access-Control-Allow-Origin", "*")]
    Swagger(app, template_file=docs_file, config=swagger_config)
    return app


def read_config(base_dir: str, path: str) -> (dict, bool):
    """
    This function reads the file containing the application config.

    Parameters
    ===========
        base_dir: str
            The base dir containing the config file passed in the second argument.
        path: str
            The input file containing the error messages

    Returns
    ========
        (dict, bool)
            if bool is False, dict will contain an error key containing the error message like:
                {'error': f"Config file not found,...."}
            if bool is True, dic will contain a dict contain the config dict.

    """
    if not os.path.isfile(path):
        return {"error": f"Config file not found, should be at {path}"}, False
    try:
        conf = json.load(open(path))
        conf["working_dir"] = os.path.join(base_dir, conf["working_dir"])
        if not os.path.exists(conf["working_dir"]):
            os.mkdir(conf["working_dir"])
        conf["model_path"] = os.path.join(base_dir, conf["model_path"]).replace("//", "")
        conf.setdefault("version", "V1")
        return conf, True
    except ValueError:
        return {"error": f"Couldn't parse config while reading file {path}"}, False


def base64_to_bytes(base64_img: str, error_code: int) -> bytes:
    """
    This function converts base64 images to an bytes file.

    Parameters
    ===========
        base64_img: str
            The base64 string representing the image.
        error_code: int
            The error code of the raised ValidationError if the string couldn't be converted.

    Returns
    ========
        bytes
            The image bytes after converting

    Raises
    =======
        Validation
            if the base64 string cannot be converted, our custom Validation error will be raised.

    """
    try:
        return base64.b64decode(base64_img)
    except binascii.Error as e:
        raise Validation(error_code) from e


def prepare_request_data(request_body: dict) -> dict:
    """
    This function prepares the request data.

    Parameters
    ===========
        request_body: dict
            Prepare the request data to work on.

    Returns
    ========
        dict
            The request data after preparing.
    """
    request_body["video"] = base64_to_bytes(request_body["video"], 4000)
    actions = set(request_body["actions"])
    if actions == set():
        raise Validation(4004)
    diff = actions - set(ACTIONS_LIST.keys())
    if diff != set():
        raise Validation(4001)
    if "actions_times" in request_body:
        if len(request_body["actions_times"]) != len(request_body["actions"]):
            raise Validation(4002)
        for action_time in request_body["actions_times"]:
            if not (
                type(action_time) == dict
                and set(action_time.keys()) == {"start", "end"}
                and type(action_time["start"]) == float
                and type(action_time["end"]) == float
            ):
                raise Validation(4003)
    return request_body


def prepare_actions(actions):
    actions_res = ActionList()
    for action in actions:
        actions_res.append(ACTIONS_LIST[action])
    return actions_res


def prepare_times(times):
    if not times:
        times = []
    times_res = TimeSlotList()
    for time_item in times:
        times_res.append(time_item)
    return times_res


def prepare_result(actions, liveness_result):
    flow_probabilities = {}
    for index, action in enumerate(actions):
        try:
            flow_probabilities[action] = liveness_result.result.actions_percentage[index]
        except IndexError as e:
            raise SomethingWrongHappened(4016) from e
    return {
        "is_live": liveness_result.result.liveness,
        "actions_percentage": flow_probabilities,
    }


def get_check_result(working_dir, model_path, body):
    video_data = body.get("video")
    actions = prepare_actions(body.get("actions"))
    time_slots = body.get("time_slots")
    # video_size = len(video_data)
    file_dir = os.path.join(working_dir, f"{uuid.uuid4()}.mp4")
    open(file_dir, "wb").write(video_data)
    times = prepare_times(time_slots)
    data = liveness_check_from_path(file_dir, model_path, actions, times)
    os.remove(file_dir)
    if data.error.code != 0:
        error = data.error.what
        code = int(data.error.code)
        print(error, code, flush=True)
        if code in ENGINE_ERRORS:
            raise SomethingWrongHappened(ENGINE_ERRORS[code])
        else:
            raise SomethingWrongHappened(4016)

    return jsonify({"result": prepare_result(body.get("actions"), data)}), 200

def get_result(file_dir, model_path,actions_list):
    actions = prepare_actions(actions_list)
    time_slots = False# [{'start':0.0,'end':0.0} for i in range(len(actions_list))]
    times = prepare_times(time_slots)
    data = liveness_check_from_path(file_dir, model_path, actions, times)
    
    #os.remove(file_dir)
    if data.error.code != 0:
        error = data.error.what
        code = int(data.error.code)
        print(error, code, flush=True)
        if code in ENGINE_ERRORS:
            raise SomethingWrongHappened(ENGINE_ERRORS[code])
        else:
            raise SomethingWrongHappened(4016)

    return prepare_result(actions_list, data)
