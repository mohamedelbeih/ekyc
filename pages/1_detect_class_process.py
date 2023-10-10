import streamlit as st
from streamlit_extras import switch_page_button
import json
from detect_class_process.encrypt import encrypt_file
import os
import requests
import base64
import subprocess , cv2
import time
import io
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="detect_classify_process",initial_sidebar_state="collapsed"
    # page_icon="👋",
)
with st.sidebar:
    if st.button("Reset",key='reset'):
        os.system("rm -r image.json")
        os.system("rm -r verify/Input/*")
        os.system("rm -r verify/selected/*")
        os.system("rm -r detect_class_process/results.json")
        os.system("rm -r Active_aliveness_verification/verified/*")
        os.system("rm -r Active_aliveness_verification/Verified_Actions/*")
        os.system("rm -r Active_aliveness_verification/input/*")

def resize(img):
    max_size = 1500.0
    h,w,c = img.shape
    max_side = max(h,w)
    ratio = max_size/max_side
    n_h = int(h * ratio)
    n_w = int(w * ratio)
    n_img = cv2.resize(img, (n_w,n_h) , interpolation = cv2.INTER_AREA)
    return n_img

st.subheader("Detecting, Classifying, Processing to extract data")
st.write("here detecting cards then classifying them then processing them to extract data")
st.write("____")
if  sum(st.session_state['documents']) >= 1:
    #preparing customized classifier
    #creating classifier configs based on selected docs
    with open("detect_class_process/card_detect_class/Config1.json") as f:
        class_configs = json.load(f)

    for i in range(len(st.session_state['documents'])-1,-1,-1) :
    
        if st.session_state['documents'][i]:
            pass
        else:
            class_configs['side_labels'].pop(i)

    with open("detect_class_process/card_detect_class/Config1_enc/Config1.json","w") as f:
        f.write(json.dumps(class_configs,indent=4))
    
    encrypt_file("detect_class_process/card_detect_class/Config1_enc/")
    os.system("cp detect_class_process/card_detect_class/Config1_enc/Config1.json detect_class_process/card_detect_class/layout_enc/Config1.json")
    os.system("cp detect_class_process/card_detect_class/Config1_enc/Config1.json detect_class_process/card_detect_class/layout_enc/Config1.json")

    # ui
    img_file = st.file_uploader("Select Image", type=["png","jpg","jpeg", 'bmp', 'tiff'], key='inf_image_path')
    no_image = False # intializing value for warning 
    if img_file != None:
        img_bytes = img_file.getvalue()
        np_image = np.array(Image.open(io.BytesIO(img_bytes))) 
        # np_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
        bgr_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        re_img = resize(bgr_img)
        _,encoded_image = cv2.imencode('.png', re_img)
        img_bytes = encoded_image.tobytes()

        st.image(img_bytes)
        if st.button("process",key="process_doc"):
            encoded_string = base64.b64encode(img_bytes)
            base64_img = encoded_string.decode('ascii')
            st.session_state['image'] = base64_img
        
            with open("image.json",'w') as f:
                f.write(json.dumps({"image":base64_img}))

            os.system("cd detect_class_process; export LD_LIBRARY_PATH=$(pwd); python e-kyc.py ../image.json")
            #subprocess.call(["cd detect_class_process; export LD_LIBRARY_PATH=$(pwd); uwsgi uwsgi.ini"], shell=True)
            #response = requests.post("http://192.168.6.43:8001/process_docs_in_image",data=detect_class_process_schema)
            
            with open('detect_class_process/results.json') as f:
                cards = json.load(f)
            
            st.session_state['detect_class_process_results'] = cards

            for card in cards:
                lbls = cards[card].keys()
                image_fields = [ x for x in lbls if "Image" in x ] 
                image_fields = image_fields + [ x for x in lbls if "image" in x ] 
                st.subheader(card +":")
                col1, col2 = st.columns(2)
                col1.write('Label')
                col2.write('Value')
                for img_field in image_fields:
                    with st.container():
                        col1, col2 = st.columns(2)
                        col1.write(img_field)
                        col2.image(base64.b64decode(cards[card][img_field].encode('utf-8')))
                    if "ersonal" in img_field:
                        with open("verify/Input/face_{}.jpg".format(card.split("_")[-1]),"wb") as f:
                            f.write(base64.b64decode(cards[card][img_field].encode('utf-8')))

                for lbl in set(lbls).difference(set(image_fields)):
                    with st.container():
                        col1,col2 = st.columns(2)
                        col1.write(lbl)
                        col2.write(cards[card][lbl])

            os.system("rm detect_class_process/results.json")

    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back2"):
            switch_page_button.switch_page("streamlit app")
    with col2:
        if st.button(label="Next",key="next2"):
            if img_file != None :
                switch_page_button.switch_page("select verification methods")
            else:
               no_image = True

    if no_image:
        st.warning("Please upload image to be processed")
else:
    st.warning("please back to the home to select the wanted documents")
    if st.button(label="Back",key="back1"):
            switch_page_button.switch_page("streamlit app")
