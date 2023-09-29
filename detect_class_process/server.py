"""
The main goal of this server is to extract Document  information.
"""
import os
import subprocess
import json
from flask import jsonify
from flask import request
from cm_engine import get_release_version as get_release_version 
from cm_engine import deinitialize_model , initialize_model
from exceptions import Validation

from utils import call_count_units, get_documents_list , base64_to_bytes
from utils import call_process_document , call_get_cards_and_recognize , process_docs_in_image
from utils import create_app
from utils import register_swagger

from validation import DocumentProcessingSchema , DocumentProcessingSchema2, NewClassifierSchema
from validation import UnitsDecrementEstimationRequest

from config import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "static", "docs.yml")

app = create_app()
app = register_swagger(app, DOCS_PATH)


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
    server_version = settings.VERSION.split("b")[0].strip('-')
    server_version_suffix = f"b{settings.VERSION.split('b')[1]}" if len(settings.VERSION.split("b")) > 1 else ""
    engine_version = get_release_version().split("b")[0].strip('-')
    engine_version_suffix = f"b{get_release_version().split('b')[1]}" if len(get_release_version().split("b")) > 1 else "0"
    return jsonify({"version": f"{server_version}-{engine_version}-{server_version_suffix}.{engine_version_suffix}".strip('-').strip('.')})


@app.route("/get_remaining_units", methods=["GET"])
def get_remaining_units_view():
    """This function is used to check the remaining units

    Returns
    =========
        flask.wrappers.Response
            A json response contains a remaining_units key
            {"remaining_units": int}
    """
    try:
        from cm_engine import get_remaining_units
        return jsonify({"remaining_units": get_remaining_units()})
    except ImportError:
        from exceptions import Validation
        raise Validation("4444")
    except Exception:
        from exceptions import InternalServerError
        raise InternalServerError("4500")

#
# @app.route("/get_estimated_units", methods=["POST"])
# def get_estimated_units_view():
#     """
#     Get estimated units view is a function that calculates the number of units,
#         which the user needs to process this task.
#     Inputs:
#         this function has no inputs.
#     Outputs:
#         number of units the user needed,
#         states code for the process.
#     """
#     data: dict = request.json
#     validated_data: dict = UnitsDecrementEstimationRequest().load(data)
#     result: dict = call_count_units(**validated_data)
#     return jsonify(result), 200


@app.route("/get_supported_documents", methods=["GET"])
def get_supported_documents_view():
    return jsonify({"supported_documents": get_documents_list()})


@app.route("/process_document", methods=["POST"])
def process_document_view():
    data: dict = request.json
    validated_data: dict = DocumentProcessingSchema().load(data)
    result: dict = call_process_document(**validated_data)
    return jsonify(result), 200

@app.route("/get_cards_and_recognize", methods=["POST"])
def get_cards_and_recognize_view():
    data: dict = request.json
    validated_data: dict = DocumentProcessingSchema().load(data)
    result: dict = call_get_cards_and_recognize(**validated_data)
    return jsonify(result), 200

@app.route("/image_preprocessor",methods=['POST'])
def get_cards_and_fields():
    data = request.json
    validated_data: dict = DocumentProcessingSchema2().load(data)
    image_base64 = validated_data['image']
    model = validated_data['document_type']
    with open("img.json","w") as f:
        f.write(json.dumps({"img":image_base64}))
    try:
        os.system( "export LD_LIBRARY_PATH=/opt/python/lib; python decode_detectron.py " + model )
        with open("labelname2img_dict.json") as f:
            labelname2img_dict = json.load(f)
        os.system("rm labelname2img_dict.json")
        return jsonify(labelname2img_dict) , 200
    except:
        raise Validation("4015")

@app.route("/process_docs_in_image", methods=["POST"])
def process_docs_in_image_view():
    data: dict = request.json
    validated_data: dict = NewClassifierSchema().load(data)
    initialize_model("card_detect_class/layout_enc","card_detect_class/recognition")
    result: dict = process_docs_in_image(**validated_data)
    deinitialize_model("card_detect_class/layout_enc","card_detect_class/recognition")
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(
        debug=int(os.getenv("FLASK_DEBUG", "0")) == 1,
        host="0.0.0.0",
        port=settings.SERVER_PORT,
        threaded=True,
    )
