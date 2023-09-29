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

from sample_utils.turn import get_ice_servers


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
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24") #480*640
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    # corner_pts = None
    # corner_pts = detect_document(img, model)
    # if corner_pts != None:# replaceing with if card is detected
    #     h_left = corner_pts[3][1] -corner_pts[0][1] 
    #     h_right = corner_pts[2][1] -corner_pts[1][1] 
    #     w_up = corner_pts[1][0] -corner_pts[0][0] 
    #     w_bottom = corner_pts[3][0] -corner_pts[2][0] 
    #     img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 0, 255), 2) 
    #     if abs(h_left - h_right) < 30 and abs(w_up-w_bottom) < 30 :
    #         img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 255, 0), 2)
    #         global height
    #         height = img.shape[0] #card height
    #         global width
    #         width = img.shape[1] # card width
    #         img = cv2.putText(img, 'Centered', (300,444), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # else:
    #     # img = cv2.polylines(img, [corner_pts.astype('int')],True, (0, 0 , 255), 2)
    #     pass
    # perform edge detection
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

RECORD_DIR = Path("Document_Aliveness_verification/Input/")

def stream_and_record():
    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
    prefix = st.session_state["prefix"]
    in_file = RECORD_DIR / f"{prefix}_input.mp4"
    # out_file = RECORD_DIR / f"{prefix}_output.flv"

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            str(in_file), format="mp4"
        )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

    # def out_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(str(out_file), format="flv")
    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        # out_recorder_factory=out_recorder_factory,
    )

if page_name in st.session_state['selected_verifications']:

    actions = st.multiselect("Pick up the verification actions",
                        ['Up','Down','Left','Right'],
                        ['Down','Left'])

    stream_and_record()
    if st.button('verify',key='doc_verf'):
        prefix = st.session_state["prefix"]
        in_file = prefix +"_input.mp4"
        os.system("cd Document_Aliveness_verification; python check_liveness.py Input/"+in_file)

        with open("Document_Aliveness_verification/Output/" + in_file.replace('.mp4','.txt')) as f:
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
        


    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back10"):
            switch_page_button.switch_page(previous_page)
    with col2:
        if st.button(label="Next",key="next6"):
            switch_page_button.switch_page(next_page)

# elif st.session_state['verifications'][3]:
#     switch_page_button.switch_page("Audio Verification")
else:
    st.warning("please back to the select verification methods page then select document aliveness verification ")
    if st.button(label="Back",key='back9'):
        switch_page_button.switch_page("select verification methods")