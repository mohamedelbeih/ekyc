#from asyncio import subprocess
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
torch.set_num_threads(12)
import json
from PIL import Image
from io import BytesIO
import base64
from config import settings
torch.set_num_threads(settings.RECOGNITION_THREADS)
import sys,os,time


mdl = "models--microsoft--trocr-small-printed/snapshots/e6eff164f5957036a190e94b1febb6d1d7f165e7"
# mdl = 'models--microsoft--trocr-base-printed/snapshots/0c708d318cd981cfc88c217fb90fb769b52ef2ff'

def recog_txt(field2_image_bas64,mdl):
    'takes mdl path , dict of fields names and base64 encodings >> return dict of fields names and text'

    processor = TrOCRProcessor.from_pretrained(mdl)
    model = VisionEncoderDecoderModel.from_pretrained(mdl)
    model.to("cpu")
    st = time.time()
    
    results = {}
    for field in field2_image_bas64:
        image = Image.open(BytesIO(base64.b64decode(field2_image_bas64[field]))).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to("cpu")
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results[field] = generated_text
    print(">"*50,time.time()-st)
    
    return results

request_time = sys.argv[1]
with open("fields_type_2_base64img"+request_time+".json") as f :
    fields_type_2_base64img =  json.load(f)  

os.system("rm fields_type_2_base64img"+request_time+".json")

results = recog_txt(fields_type_2_base64img,mdl)

with open("fields_type_2_text"+request_time+".json","w") as g:
    g.write(json.dumps(results,indent=4))
    

