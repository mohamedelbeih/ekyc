import streamlit as st
from streamlit_extras import switch_page_button
import os
from streamlit_webrtc import webrtc_streamer
import av , sys
import cv2 , os , math
import time
import uuid,json , threading
import uuid
from pathlib import Path

import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.turn import get_ice_servers
lock = threading.Lock()
st.set_page_config(
    page_title="Active Aliveness Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)


# st.sidebar.success("Select a demo above.")
st.subheader("Active Aliveness Verification through camera")
st.write("here Active Aliveness Verification between previoulsy given image and live stream from camera , delect wanted actions to verify then stream performing these actions")
st.write("____")

page_name = 'Active Aliveness Verification'
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
#verf_pages = ['Face Verification','Active Aliveness Verification','Document Aliveness Verification','Audio Verification']
# def get_start_time():
#     global prefix
#     prefix = time.time()
#     st.session_state['stream_starting_time'] = time.time()
#     print("<><><>",st.session_state['stream_starting_time'])

def check_aliveness(current_action):
    os.system("cd Active_aliveness_verification; python get_video.py " + current_action)
    os.system("cd Active_aliveness_verification; export LD_LIBRARY_PATH=$(pwd); python e-kyc.py video.avi " + current_action)
count = 0
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # perform edge detection
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    # print("<><><>",st.session_state['stream_starting_time'])

    # print(st.session_state['stream_starting_time'])
    if os.listdir("Active_aliveness_verification/start_time") == []:
        start_time = str(time.time())
        os.makedirs("Active_aliveness_verification/start_time/"+start_time)
    else:
        start_time = os.listdir("Active_aliveness_verification/start_time")[0]
    current_action_idx = math.floor(( time.time() - float(start_time) ) / 10 )
    print("<<<<<",current_action_idx)


    current_action = actions[current_action_idx]
    print(">>>>",current_action)


    os.makedirs("Active_aliveness_verification/actions/" + current_action,exist_ok=True)

    images_num = len(os.listdir("Active_aliveness_verification/actions/"+ current_action))
    print("-----------------------------")
    if  images_num == 30 :
        #making video and path it to get results
        print("cheeeeeeeeeeeeeeeeeeeeeecking",current_action)
        with lock:
            check_aliveness(current_action)
        if False:
            t2 = threading.Thread(check_aliveness,args=(current_action))
            t2.start()         

    if images_num <= 30:     
        cv2.imwrite("Active_aliveness_verification/actions/"+ current_action + "/" + str(time.time()) + ".jpg" , img)   
        print("saving",current_action) 

    try:
        print("trying")
        with open("Active_aliveness_verification/actions/"+ current_action + "/results.json") as f:
            # print("OOOOOO")
            res = json.load(f)

        # while True:
        if not res["is_live"] :
            # print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
            # make new folder with results lw fe file 2bl time.time() b 2 secs msln
            # print(actions[15]) # to break the call back

            return av.VideoFrame.from_ndarray(img.fill(0), format="bgr24")
            
    except:
        pass


    return av.VideoFrame.from_ndarray(img, format="bgr24")

# RECORD_DIR = Path("./Active_aliveness_verification/Input/")

def stream_and_record():
    # in_file = RECORD_DIR / f"{prefix}_input.mp4"
    # out_file = RECORD_DIR / f"{prefix}_output.flv"

    # def in_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(
    #         str(in_file), format="mp4"
    #     )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

    # def out_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(str(out_file), format="flv")

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        # rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        # video_frame_callback=video_frame_callback,
        video_frame_callback=video_frame_callback,
        async_processing=False
        # on_change=get_start_time
        # in_recorder_factory=in_recorder_factory,
        # out_recorder_factory=out_recorder_factory,
    )


if page_name in st.session_state['selected_verifications']:
    actions = st.multiselect("Pick up the verification actions",
                       ['Up','Down','Left','Right','Smiling','Blinking'],
                       ['Down','Smiling','Blinking'])
    os.system("rm -r Active_aliveness_verification/actions/*")
    os.system("rm -r Active_aliveness_verification/start_time/*")

    for a in actions:
        os.makedirs("Active_aliveness_verification/actions/" + a,exist_ok=True)
    
    # record video and convert it to mp4 in Active_aliveness_verification/uploads/video.mp4
    
    st.write("\n\n")
    # t1 = threading.Thread(stream_and_record,args=())
    # t1.start()
    stream_and_record() 
  
    st.write("\n\n")
    if st.button("verify"):
        os.system("cd Active_aliveness_verification; export LD_LIBRARY_PATH=$(pwd); python e-kyc.py "+st.session_state["prefix"] +"_input.mp4 " + "_".join(actions))
        with open("Active_aliveness_verification/results.json") as f:
            results = json.load(f)

        if results['is_live']:
            st.success("is live")
        else:
            st.error('is_not_live')

        st.subheader("Actions percentage:")
        for k,v in results['actions_percentage'].items():
            if v > 50:
                st.success(k + " verified with percentage " + str(v))
            else:
                st.warning(k + " verified with percentage " + str(v))
      
        
    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back8"):
            switch_page_button.switch_page(previous_page)
    with col2:
        if st.button(label="Next",key="next5"):
            switch_page_button.switch_page(next_page)
else:
    st.warning("please back to the select verification methods page then select active aliveness verification ")
    if st.button(label="Back",key='back7'):
        switch_page_button.switch_page("select verification methods")

