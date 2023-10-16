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
from check_liveness import check_tilting
from sample_utils.turn import get_ice_servers
from inference import *
st.set_page_config(
    page_title="Document Aliveness Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.subheader("Document Aliveness Verification through camera")
st.write("here Verification between previoulsy given image and live stream from card")
st.write("____")

page_name = 'Document Aliveness Verification'
count = 0
#load detection model
model = initialze_scripted_model('/media/asr7/19a7589f-b05f-4923-9b94-5803bff11012/OCR/e-kyc-pipeline/Document_Aliveness_verification/v.10/model.ts')
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24") #480*640
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    img = cv2.rectangle(img,(50,115),(590,460),(255,0,0))

    corner_pts = None
    card , corner_pts = detect_document(img, model)
    verf_Actions = os.listdir("Verified_Actions")
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
        print(">>>>>>>>>>>>>",tilting,h_left,h_right,w_up,w_bottom)
        if tilting and (tilting in actions):
            os.makedirs('Verified_Actions/'+tilting,exist_ok=True)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


actions = st.multiselect("Pick up the verification actions",
                       ['Up','Down','Left','Right'],
                       ['Down','Left'])

st.info('Please stream with your card almost filling the blue rectangle then start titlting as required')
webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True,"audio": False,},async_processing=True,rtc_configuration={"iceServers": get_ice_servers()},)

if st.button('verify',key='doc_verf'):
    prefix = st.session_state["prefix"]
    in_file = prefix +"_input.mp4"
    os.system("python check_liveness.py Input/"+in_file)

    with open("Output/" + in_file.replace('.mp4','.txt')) as f:
        results = f.read()
    if len(results.split("_")) > 0:
        verified_selected_actions = set(actions).intersection(set(results.split("_")))
        unverified_selected_actions = set(actions).difference(set(results.split("_")))
        for d in verified_selected_actions:
            st.success(d + " rotation verified")
        for d in unverified_selected_actions:
            st.warning(d + " rotation couldn't be verified")
    else:
        st.error("Couldn't verify any action of the selected actions")
    
