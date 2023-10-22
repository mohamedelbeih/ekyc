import streamlit as st
from streamlit_extras import switch_page_button
import base64
import json
import face_recognition
import os , cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import json , cv2 , io

st.set_page_config(
    page_title="Face Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.subheader("Face verification through camera")
st.write("here verifying detected face in the previous given image with a live image from camera ( real person or another image )")
st.write("____")

with st.sidebar:
    if st.button("Reset",key='reset'):
        os.system("rm -r Data/Matched_faces/"+st.session_state['user_id']+"/*")
        os.system("rm -r Data/detected_faces_in_verf_image/"+st.session_state['user_id']+"/*")         
        os.system("rm -r Data/selected_faces/"+st.session_state['user_id']+"/*")

def resize(img):
    max_size = 1500.0
    h,w,c = img.shape
    max_side = max(h,w)
    ratio = max_size/max_side
    n_h = int(h * ratio)
    n_w = int(w * ratio)
    n_img = cv2.resize(img, (n_w,n_h) , interpolation = cv2.INTER_AREA)
    return n_img

page_name = 'Face Verification'
try:
    page_idx_in_selected = st.session_state['selected_verifications'].index(page_name)
    if len(st.session_state['selected_verifications']) > 1:
        next_page =  st.session_state['selected_verifications'][page_idx_in_selected+1]
    else:
        next_page = "end of verifications"
except:
    pass

if "selected_verifications" in st.session_state:
    if page_name in st.session_state['selected_verifications']:
      st.write("Detected Faces in the previously given image :")
      faces = os.listdir("Data/faces/"+st.session_state['user_id'])
      os.makedirs("Data/selected_faces/"+ st.session_state['user_id'] ,exist_ok=True)
      for face in faces:
          img= Image.open("Data/faces/"+st.session_state['user_id']+"/"+face)
          with st.container():
              col1, col2  = st.columns([0.5,0.5],gap="large")
              with col1:
                  st.image(img)
              with col2:
                  if st.checkbox("Personal_image_"+face.split("_")[-1],value=True):
                      os.system("cp Data/faces/"+st.session_state['user_id']+"/"+face + " Data/selected_faces/"+st.session_state['user_id']+"/"+face)
                  else:
                      os.system("rm Data/selected_faces/"+st.session_state['user_id']+"/"+face)

      uploaded_picture = st.file_uploader("upload Image", type=["png","jpg","jpeg", 'bmp', 'tiff'], key='face_imge')
      captured_picture = st.camera_input("Or Take a picture")
      procesing_done , matched = False , 0
      if captured_picture or uploaded_picture:
          picture = captured_picture if captured_picture else uploaded_picture
          st.session_state['face_verf_pic'] = picture
          st.write("Captured image")
          st.image(picture)
          img_bytes = picture.getvalue()
          np_image = np.array(Image.open(io.BytesIO(img_bytes))) 
          # np_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
          rgb_img = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

          with st.spinner('Please wait .. Detecting faces in image'):
              selected_faces = os.listdir("Data/selected_faces/"+st.session_state['user_id'])
              selected_faces = [ "Data/selected_faces/"+st.session_state['user_id']+"/" + x for x in selected_faces]
              selected_faces_encodes = []
              for f in selected_faces:
                  img = face_recognition.load_image_file(f)
                  # face = face_recognition.face_locations(img_modi)[0]
                  encode = face_recognition.face_encodings(img)[0]
                  selected_faces_encodes.append(encode)
                  
              # img_modi = cv2.cvtColor(encoded_image,cv2.COLOR_BGR2RGB)
              detected_faces = face_recognition.face_locations(rgb_img)
              if len(detected_faces) > 0:
                  st.subheader('Detected faces :')
              counter = 0
              os.makedirs("Data/detected_faces_in_verf_image/"+st.session_state['user_id'],exist_ok=True)
              for f in detected_faces:
                  face = np_image[f[0]:f[2],f[3]:f[1]]
                  cv2.imwrite('Data/detected_faces_in_verf_image/'+st.session_state['user_id']+"/"+str(counter)+".jpg",cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
                  st.image(face)
                  counter += 1

              encodes = face_recognition.face_encodings(rgb_img)
              st.subheader("Matched faces : ")
              for i in range(len(detected_faces)):
                  try:
                      with st.container():
                          col1, col2 ,col3  = st.columns([0.35,0.3,0.35],gap="large")                
                          values = face_recognition.compare_faces(encodes,selected_faces_encodes[i])
                          idx = [ values.index(x) for x in values if x]
                          idx = idx[0]
                          croped_face = np_image[detected_faces[idx][0]:detected_faces[idx][2],detected_faces[idx][3]:detected_faces[idx][1]]
                          matched =  matched +1 if sum(values) > 0 else matched 
                          with col1:
                              st.image(croped_face)
                          with col2:
                              st.write("is matching with")
                          with col3:
                              st.image(selected_faces[i])
                      os.makedirs('Data/Matched_faces/'+st.session_state['user_id']+"/"+str(i),exist_ok=True)
                      os.system('cp '+ selected_faces[i] +' Data/Matched_faces/'+st.session_state['user_id']+"/"+str(i)+"/face.jpg")
                      cv2.imwrite('Data/Matched_faces/'+st.session_state['user_id']+"/"+str(i)+"/with_face.jpg",
                                  cv2.cvtColor(croped_face,cv2.COLOR_BGR2RGB))
                  except:
                      pass
              procesing_done = True
          if len(detected_faces) == 0:
              st.warning("couldn't detect any faces in the given image, please try again")
          if matched == 0 and procesing_done:
              st.error("Couldn't match any faces, please try again")
              
      col1, col2  = st.columns([0.85,0.15],gap="large")
      with col1:
          if st.button(label="Back",key="back6"):
              switch_page_button.switch_page("select verification methods")
      with col2:
          no_match = True if matched == 0 else False
          if st.button(label="Next",key="next4",disabled=no_match):
              switch_page_button.switch_page(next_page)
              

    elif st.session_state['verifications'][1]:
        switch_page_button.switch_page("Active Aliveness Verification")
else:
    st.warning("please back to the select verification methods page then select face verification")
    if st.button(label="Back",key='back5'):
        switch_page_button.switch_page("select verification methods")
