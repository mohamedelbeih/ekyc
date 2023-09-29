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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def recog_txt(field):
    ''' field and base64img , append text result to results list '''
    # field_key,field_value = field
    field_key, field_value = field
    image = Image.open(BytesIO(base64.b64decode(field_value))).convert("RGB")
    mdl_path = "models--microsoft--trocr-small-printed/snapshots/e6eff164f5957036a190e94b1febb6d1d7f165e7"
    # mdl_path = 'models--microsoft--trocr-base-printed/snapshots/0c708d318cd981cfc88c217fb90fb769b52ef2ff'
    processor = TrOCRProcessor.from_pretrained(mdl_path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to("cpu")
    model = VisionEncoderDecoderModel.from_pretrained(mdl_path)
    model.to("cpu")
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    with open("fields/"+field_key,"w") as f:
        f.write(generated_text)

request_time = sys.argv[1]
with open("fields_type_2_base64img"+request_time+".json") as f :
    fields_type_2_base64img =  json.load(f)  
os.system("rm fields_type_2_base64img"+request_time+".json")

keys_values = list(fields_type_2_base64img.items())
keys = [ x[0] for x in keys_values]
values = [ x[1] for x in keys_values]
results = {}
st = time.time()

p=mp.Pool(4)
out=list(p.map(recog_txt,keys_values))
print(">"*50,time.time()-st)

with open("fields_type_2_text"+request_time+".json","w") as g:
    g.write(json.dumps(results,indent=4))
