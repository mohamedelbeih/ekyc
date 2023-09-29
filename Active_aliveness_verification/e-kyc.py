from error_checks import validate_request_data
from utils import prepare_request_data , get_result
import json , sys
import time
def liveness_check(file_name,actions_list):
    return get_result("Input/"+file_name, "./model_enc/", actions_list)


#{'is_live': False, 'actions_percentage': {'Down': 100.0, 'Up': 0.0}}
actions = sys.argv[2].split("_")
file_name = sys.argv[1]
with open("results.json",'w') as f:
    ts = time.time()
    f.write(json.dumps(liveness_check(file_name,actions)))
    te = time.time()
    print(te-ts)

