# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:00:24 2022

@author: ali ramadan
"""

import requests
import base64

def base64_encode(file_name):
    with open(file_name, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    return ""
def run(path):
    url = "http://0.0.0.0:8000/liveness_check"
    v_path=path+"/1.mp4"
    actions_path=path+"/actions.txt"
    v=base64_encode(v_path)
    actions=open(actions_path).read().splitlines()
    actions_time=[{"start":0.0,"end":0.0},{"start":0.0,"end":0.0},{"start":0.0,"end":0.0},{"start":0.0,"end":0.0}]
    payload={
            "video": v,
             "actions": actions,
             "actions_times":actions_time
            }
    
    response = requests.request("POST", url, json=payload)
    print(response.text,response.status_code)
import sys

run(sys.argv[1])
