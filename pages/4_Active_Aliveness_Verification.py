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
from PIL import Image
import cv2
import face_recognition
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.turn import get_ice_servers
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 , os
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from sample_utils.turn import get_ice_servers
import av

lock = threading.Lock()
st.set_page_config(
    page_title="Active Aliveness Verification",initial_sidebar_state="collapsed"
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

def blend(img): 

    fraction = 0.3

    output = np.full_like(img,0)
    output[:,:] = [0,0,100]

    output = output*fraction + img
    output = output.astype(np.uint8)
    return output

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)

    if len(os.listdir("Active_aliveness_verification/verified")) > 5:
        if detection_result.face_landmarks == []:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            verf_Actions = os.listdir("Active_aliveness_verification/Verified_Actions")
            need_actions = list(set(selected_actions).difference(set(verf_Actions)))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            t =  "please verify : " + need_actions[0] if len(need_actions) > 0 else "Done verifying all of the selected actions"
            annotated_image = cv2.putText(annotated_image,t , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2, cv2.LINE_AA)
            #verified
            t =  "actions verified till now : " + ",".join(verf_Actions)
            annotated_image = cv2.putText(annotated_image,t , (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            #needed_to verify
            t =  "actions still unverified : " + ",".join(need_actions)
            annotated_image = cv2.putText(annotated_image,t , (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            blend2score = { detection_result.face_blendshapes[0][i].category_name : detection_result.face_blendshapes[0][i].score for i in range(len(detection_result.face_blendshapes[0]))}
            # face_blends = sorted(face_blends,reverse=True)
            if len(need_actions) > 0:
                if blend2score[need_actions[0]] > 0.5:
                    os.makedirs("Active_aliveness_verification/Verified_Actions/"+need_actions[0],exist_ok=True)
               
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
    else:
        try:
            rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_encode = face_recognition.face_encodings(rgb)[0]
            values = face_recognition.compare_faces([encode],img_encode)
            if sum(values) > 0:
                os.makedirs("Active_aliveness_verification/verified/"+str(time.time()))
            else:
                img = cv2.rectangle(img, (100,350), (540,450), (0,0,255), -1) 
                img = cv2.putText(img, "Selected person is not streaming", (120,410), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")
        except:
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image


if page_name in st.session_state['selected_verifications']:
    not_matched = True
    faces = os.listdir("verify/Input")
    selected_faces = 0
    for face in faces:
        img= Image.open("verify/Input/"+face)
        with st.container():
            col1, col2  = st.columns([0.5,0.5],gap="large")
            with col1:
                st.image(img)
            with col2:
                if st.checkbox("Personal_image_"+face.split("_")[-1],value=False):
                    os.system("cp verify/Input/"+face + " Active_aliveness_verification/input/"+face)
                    selected_faces +=1
                    idx = faces.index(face)
                else:
                    os.system("rm Active_aliveness_verification/input/"+face)

    base_options = python.BaseOptions(model_asset_path='Active_aliveness_verification/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)
    if selected_faces == 1 :
        img= face_recognition.load_image_file("verify/Input/"+faces[idx])
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(rgb)[0]
        #st.info("you will need to verify at least 3 live actions to continue verification")
        if st.checkbox('test any person aliveness actions',value=False):
            for i in range(6):
                os.makedirs("Active_aliveness_verification/verified/"+str(i),exist_ok=True)
        else:
            os.system("rm -r Active_aliveness_verification/verified/*")

        actions = ['mouthPucker', 'browInnerUp', 'eyeSquintRight', 'eyeSquintLeft', 'browOuterUpLeft', 'eyeBlinkRight', 
                'eyeBlinkLeft', 'mouthRollLower', 'eyeLookDownRight', 'eyeLookDownLeft', 'browOuterUpRight', 'mouthShrugLower', 
                'eyeLookUpLeft', 'eyeLookOutLeft', 'eyeLookUpRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookInLeft', 'mouthRollUpper',
                'jawOpen', 'jawLeft', 'mouthShrugUpper', 'mouthPressRight', 'mouthFrownRight', 'browDownRight', 'browDownLeft', 'mouthFrownLeft',
                'eyeWideLeft', 'mouthRight', 'mouthPressLeft', 'mouthClose', 'eyeWideRight', 'mouthLowerDownRight', 'mouthFunnel', 'mouthLowerDownLeft',
                'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchRight', 'mouthStretchLeft', 'mouthLeft', 'mouthUpperUpRight', 'mouthUpperUpLeft', 'jawRight', 
                'jawForward', 'cheekPuff', 'noseSneerLeft', 'mouthSmileLeft',
                '_neutral', 'cheekSquintLeft', 'mouthSmileRight', 'noseSneerRight', 'cheekSquintRight']
        selected_actions = st.multiselect("Select actions to do in the stream",actions,['eyeBlinkRight','mouthSmileLeft','jawOpen'])

        webrtc_streamer(key="example", video_frame_callback=callback, media_stream_constraints={"video": True,"audio": False,},async_processing=True,rtc_configuration={"iceServers": get_ice_servers()},)
        if set(os.listdir("Active_aliveness_verification/Verified_Actions")).difference(set(actions)) == set():
            not_matched = False
            os.system("rm -r Active_aliveness_verification/Verified_Actions/*")
    else:
        st.warning("you should select 1 face only to verify aliveness")

    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back8"):
            switch_page_button.switch_page(previous_page)
    with col2:
        if st.button(label="Next",key="next5",disabled=not_matched):
            switch_page_button.switch_page(next_page)
        
else:
    st.warning("please back to the select verification methods page then select active aliveness verification ")
    if st.button(label="Back",key='back7'):
        switch_page_button.switch_page("select verification methods")

