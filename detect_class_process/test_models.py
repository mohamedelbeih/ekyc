from cm_engine import *
import time
with open("layouts") as f:
    layouts = f.readlines()
    layouts = [ x.strip() for x in layouts ]
    
    
with open("recogs") as f:
    recogs = f.readlines()
    recogs = [ x.strip() for x in recogs ]
    
st = time.time()    
for i in range(51):
    print(i , layouts[i].split("/")[-2])
    initialize_model(layouts[i],"/media/asr7/19a7589f-b05f-4923-9b94-5803bff11012/Cluster/OCR/document-processing-server/card_detector/recognition" )
    #deinitialize_model(layouts[i] ,recogs[i])

print(">>>>>>>>>>>>>>>>>>>>>>>>" , time.time() - st )
time.sleep(5000)
