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




def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)

    if detection_result.face_landmarks == []:
       return av.VideoFrame.from_ndarray(img, format="bgr24")
    else:
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        face_blends = [(detection_result.face_blendshapes[0][i].score , detection_result.face_blendshapes[0][i].category_name) for i in range(len(detection_result.face_blendshapes[0]))]
        face_blends = sorted(face_blends,reverse=True)
        for i in range(2):
            blendshape = face_blends[i][1]
            blendscore = str(int(face_blends[i][0]*100))
            x , y = 10 , 400 + 30*(i+1)
            annotated_image = cv2.putText(annotated_image, blendshape +" "+blendscore+"%", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if "Smile" in blendshape:
                os.makedirs("Verified_Actions/Smile",exist_ok=True)
                #wanna stop stream if os.listdir(Verified_Actions) == actions
        
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")        
       

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

    base_options = python.BaseOptions(model_asset_path='Active_aliveness_verification/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    webrtc_streamer(key="example", video_frame_callback=callback, media_stream_constraints={"video": True,"audio": False,},async_processing=True,rtc_configuration={"iceServers": get_ice_servers()},)


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

