from utils import *
import sys
import json
from cm_engine import get_release_version as get_release_version 
from cm_engine import deinitialize_model , initialize_model
from exceptions import Validation
from validation import DocumentProcessingSchema , DocumentProcessingSchema2, NewClassifierSchema
from validation import UnitsDecrementEstimationRequest
from config import settings
user_id = sys.argv[1]

def detect_class_process(json_file_path,user_id):
    with open(json_file_path) as f:
        data = json.load(f)
    data['return_cropped_images'] = False
    data['k2_recognition'] = False
    validated_data: dict = NewClassifierSchema().load(data)
    initialize_model("card_detect_class/"+user_id,"card_detect_class/recognition")
    result: dict = process_docs_in_image(user_id,**validated_data)
    deinitialize_model("card_detect_class/"+user_id,"card_detect_class/recognition")
    return result
    
results = detect_class_process(sys.argv[2],user_id)
with open(user_id+"_results.json","w") as f:
    f.write(json.dumps(results))
