#from asyncio import subprocess
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
from PIL import Image
from io import BytesIO
import base64
from config import settings
import sys,os,time
import threading

def recog_txt(field_name,field_value):
    ''' field and base64img , append text result to results list '''
    image = Image.open(BytesIO(base64.b64decode(field_value))).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to("cpu")
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    breakpoint()
    results[field_name] = generated_text

request_time = sys.argv[1]
with open("fields_type_2_base64img"+request_time+".json") as f :
    fields_type_2_base64img =  json.load(f)  
os.system("rm fields_type_2_base64img"+request_time+".json")

mdl_path = "models--microsoft--trocr-small-printed/snapshots/e6eff164f5957036a190e94b1febb6d1d7f165e7"
# mdl = 'models--microsoft--trocr-base-printed/snapshots/0c708d318cd981cfc88c217fb90fb769b52ef2ff'
processor = TrOCRProcessor.from_pretrained(mdl_path)
model = VisionEncoderDecoderModel.from_pretrained(mdl_path)
model.to("cpu")

results = {}
threads = []
for k,v in fields_type_2_base64img.items():
    t = threading.Thread(target=recog_txt, args=(k,v))
    t.start()
    threads.append(t)
for t in threads:
    t.join()

with open("fields_type_2_text"+request_time+".json","w") as g:
    g.write(json.dumps(results,indent=4))
