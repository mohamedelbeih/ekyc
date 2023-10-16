import streamlit as st
from streamlit_extras import switch_page_button
import os
from streamlit_webrtc import webrtc_streamer
import av
import cv2 , os
import time
import uuid,json
import uuid
from pathlib import Path
import numpy as np
import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from Document_Aliveness_verification.check_liveness import check_tilting
from sample_utils.turn import get_ice_servers
from Document_Aliveness_verification.inference import *
import face_recognition
from sample_utils.turn import get_ice_servers
from PIL import Image


st.set_page_config(
    page_title="Document Aliveness Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.subheader("Document Aliveness Verification through camera")
st.write("here Verification between previoulsy given image and live stream from card")
st.write("____")

page_name = 'Document Aliveness Verification'
try:
    page_idx_in_selected = st.session_state['selected_verifications'].index(page_name)

    if page_idx_in_selected == 0:
        previous_page = 'select verification methods'

    if (len(st.session_state['selected_verifications']) > 1 ) and (page_idx_in_selected > 0):
        previous_page = st.session_state['selected_verifications'][page_idx_in_selected-1]

    if len(st.session_state['selected_verifications'])-1 > page_idx_in_selected :
        next_page =  st.session_state['selected_verifications'][page_idx_in_selected+1]

    if page_idx_in_selected == len(st.session_state['selected_verifications'])-1 :
        next_page = "end of verifications"

except:
    pass

#load detection model
model = initialze_scripted_model('/media/asr7/19a7589f-b05f-4923-9b94-5803bff11012/OCR/e-kyc-pipeline/Document_Aliveness_verification/v.10/model.ts')
class_model = initialze_scripted_model('/media/asr7/19a7589f-b05f-4923-9b94-5803bff11012/OCR/e-kyc-pipeline/Document_Aliveness_verification/card_classifier/model.ts')
user_id = st.session_state['user_id']
os.makedirs("Document_Aliveness_verification/Verified_Actions/"+user_id,exist_ok=True)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24") #480*640
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    img = cv2.rectangle(image,(50,115),(590,460),(255,0,0))
    corner_pts = None

    card , corner_pts = detect_document(img, model)
    verf_Actions = os.listdir("Document_Aliveness_verification/Verified_Actions/"+user_id)
    need_actions = list(set(actions).difference(set(verf_Actions)))
    t =  "please verify : " + need_actions[0] if len(need_actions) > 0 else "Done verifying all of the selected actions"
    img = cv2.putText(img,t , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2, cv2.LINE_AA)
    #verified
    t =  "actions verified till now : " + ",".join(verf_Actions)
    img = cv2.putText(img,t , (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    #needed_to verify
    t =  "actions still unverified : " + ",".join(need_actions)
    img = cv2.putText(img,t , (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    if card:# replaceing with if card is detected
        h_left = corner_pts[3][1] -corner_pts[0][1] 
        h_right = corner_pts[2][1] -corner_pts[1][1] 
        w_up = corner_pts[1][0] -corner_pts[0][0] 
        w_bottom = corner_pts[2][0] -corner_pts[3][0] 
        tilting = check_tilting(30,None, None,h_left,h_right,w_bottom,w_up)
        if tilting and (tilting in actions):
            x1,x2 = min([corner_pts[0][0],corner_pts[1][0],corner_pts[2][0],corner_pts[3][0]]) , max([corner_pts[0][0],corner_pts[1][0],corner_pts[2][0],corner_pts[3][0]])
            y1,y2 = min([corner_pts[0][1],corner_pts[1][1],corner_pts[2][1],corner_pts[3][1]]) , max([corner_pts[0][1],corner_pts[1][1],corner_pts[2][1],corner_pts[3][1]])
            doc_type = class_doc(image[int(y1):int(y2),int(x1):int(x2)],class_model)
            doc_type_verf = True if selected_doc_type == doc_type else False
            rgb = cv2.cvtColor(image[int(y1):int(y2),int(x1):int(x2)],cv2.COLOR_BGR2RGB)
            img_encode = face_recognition.face_encodings(rgb)[0]
            values = face_recognition.compare_faces([encode],img_encode)
            doc_personal_img_verf = True if sum(values) > 0 else False
            print(doc_type_verf,doc_type,values)
            if doc_personal_img_verf and doc_type_verf:
                os.makedirs('Document_Aliveness_verification/Verified_Actions/'+user_id+"/"+tilting,exist_ok=True)
            else:
                img = cv2.putText(img,"verified movement is done but not the selected card" , (473,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

if "selected_verifications" in st.session_state:
    if page_name in st.session_state['selected_verifications']:
        actions = st.multiselect("Pick up the verification actions",
                            ['Up','Down','Left','Right'],
                            ['Down','Left'])

        cards = os.listdir("Document_Aliveness_verification/cards/"+user_id)
        selected_cards = 0
        for c in cards:
            img= Image.open("Document_Aliveness_verification/cards/"+user_id+"/"+c)
            with st.container():
                col1, col2  = st.columns([0.5,0.5],gap="large")
                with col1:
                    st.image(img)
                with col2:
                    if st.checkbox(c,value=False):
                        selected_cards +=1
                        idx = cards.index(c)        

        if selected_cards == 1:
            with open("detect_class_process/"+user_id+"_results.json") as f:
                cards_res = json.load(f)
                selected_doc_type = cards_res["card_"+str(idx+1)]['doc_type'] 
            img= face_recognition.load_image_file("Document_Aliveness_verification/cards/"+user_id+"/"+cards[idx])
            encode = face_recognition.face_encodings(img)[0]
            webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True,"audio": False,},
                            async_processing=True,rtc_configuration={"iceServers": get_ice_servers()},)
            st.info('Please stream with your card almost filling the blue rectangle then start titlting as required')
        else:
            st.warning("you should select only 1 document to verify document aliveness")

    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back10"):
            switch_page_button.switch_page(previous_page)
    with col2:
        if st.button(label="Next",key="next6"):
            switch_page_button.switch_page(next_page)
else:
    st.warning("please back to the select verification methods page then select document aliveness verification ")
    if st.button(label="Back",key='back9'):
        switch_page_button.switch_page("select verification methods")