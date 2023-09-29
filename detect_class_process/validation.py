import json
import os

from marshmallow import EXCLUDE, validate
from marshmallow import fields
from marshmallow import pre_load
from marshmallow import Schema
from marshmallow import validates

from exceptions import Validation
from utils import get_documents_list


class UnitsDecrementEstimationRequest(Schema):
    """
        Units Decrement Estimation Request Object,
        which will contain the rectified data.
    """

    class Meta:
        unknown = EXCLUDE

    # TODO: Populate this schema


class DocumentProcessingSchema(Schema):
    """
        DocumentProcessing Schema Object,
        which will contain the rectified data.
    """
    class Meta:
        unknown = EXCLUDE

    image = fields.String(required=True, error_messages={"required": "4000", "invalid": "4001"})
    return_cropped_images = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4002"})
    # enable_double_inference = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4003"})
    document_type = fields.String(
        required=True,
        error_messages={"invalid": "4005", "required": "4004"},
    )

    @validates("document_type")
    def validate_document_type(self, value):
        if value not in get_documents_list():
            raise Validation("4005")

class DocumentProcessingSchema2(Schema):
    """
        DocumentProcessing Schema Object,
        which will contain the rectified data.
    """
    class Meta:
        unknown = EXCLUDE

    image = fields.String(required=True, error_messages={"required": "4000", "invalid": "4001"})
    return_cropped_images = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4002"})
    # enable_double_inference = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4003"})
    document_type = fields.String(
        required=True,
        error_messages={"invalid": "4005", "required": "4004"},
    )

    @validates("document_type")
    def validate_document_type(self, value):
        if value not in os.listdir("detectron_model"):
            raise Validation("4005")

class NewClassifierSchema(Schema):
    """
        DocumentProcessing Schema Object,
        which will contain the rectified data.
    """
    class Meta:
        unknown = EXCLUDE

    image = fields.String(required=True, error_messages={"required": "4000", "invalid": "4001"})
    return_cropped_images = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4002"})
    k2_recognition = fields.Boolean(required=False, load_default=True, error_messages={"invalid": "4003"})