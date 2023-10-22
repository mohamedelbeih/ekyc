import streamlit as st
from streamlit_extras import switch_page_button
import os
import cv2 , base64 , json
import numpy as np

def get_cv2img_from_base64(base64_string):
    img = base64.b64decode(base64_string.encode('utf-8'))
    #reading image from bytes
    img = np.frombuffer(img, dtype=np.uint8)
    #converting to np data structure
    img = cv2.imdecode(img, flags=1)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

st.set_page_config(
    page_title="Audio Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.success("Your verifications have been done")
st.write("________")

st.markdown("<h1 style='text-align: center; color: grey;'>Summary</h1>",
             unsafe_allow_html=True)

st.header("Processing")
st.subheader('The given image : ')
st.image(get_cv2img_from_base64(st.session_state['image']))
st.subheader('processing results:')
with open('Data/Processing_results/'+st.session_state['user_id']+'.json') as f:
    cards = json.load(f)
for card in cards:
    lbls = cards[card].keys()
    image_fields = [ x for x in lbls if "Image" in x ] 
    image_fields = image_fields + [ x for x in lbls if "image" in x ] 
    st.markdown("**"+card +":"+"**")
    col1, col2 = st.columns(2)
    col1.write('Label')
    col2.write('Value')
    for img_field in image_fields:
        with st.container():
            col1, col2 = st.columns(2)
            col1.write(img_field)
            col2.image(base64.b64decode(cards[card][img_field].encode('utf-8')))
    for lbl in set(lbls).difference(set(image_fields)):
        with st.container():
            col1,col2 = st.columns(2)
            col1.write(lbl)
            col2.write(cards[card][lbl])
##########################################
faces_path = 'Data/faces/'+st.session_state['user_id']+"/"
if 'Face Verification' in st.session_state['selected_verifications']:
    faces = os.listdir(faces_path)
    if len(faces) > 0:
        st.write("________")
        st.header('Face verification')
        st.subheader('image to be verified')
        st.image(st.session_state['face_verf_pic'])
        st.subheader("Extracted faces from document processing")
        
        cols = st.columns(len(faces))
        for i in range(len(cols)):
            cols[i].image(faces_path+faces[i])
        st.subheader('Detected faces in image to be verified')
        Detected_faces_path = 'Data/detected_faces_in_verf_image/'+st.session_state['user_id']+"/"
        faces = os.listdir(Detected_faces_path)
        cols = st.columns(len(faces))
        for i in range(len(cols)):
            cols[i].image(Detected_faces_path+faces[i])

        st.subheader('Matched Faces :')
        for x in os.listdir('Data/Matched_faces/'+st.session_state['user_id']):
            with st.container():
                col1,col2 = st.columns(2)
                col1.image('Data/Matched_faces/'+st.session_state['user_id']+"/"+x+"/face.jpg")
                col2.image('Data/Matched_faces/'+st.session_state['user_id']+"/"+x+"/with_face.jpg")

##########################################
faces_path = 'Data/Active_aliveness_selected_face/'+st.session_state['user_id']
if 'Active Aliveness Verification' in st.session_state['selected_verifications']:
    face = os.listdir('Data/Active_aliveness_selected_face/'+st.session_state['user_id'])
    if len(face) > 0:
        st.write("________")
        st.header('Active Aliveness')
        st.subheader("Selected face")
        st.image('Data/Active_aliveness_selected_face/'+st.session_state['user_id']+"/"+face[0])
        st.subheader("Verified Actions")
        verf_actions = os.listdir('Data/Active_Aliveness_Verified_Actions/'+st.session_state['user_id'])
        for i in range(len(verf_actions)):
            st.write(verf_actions[i].replace(".jpg",""))
            st.image('Data/Active_Aliveness_Verified_Actions/'+st.session_state['user_id']+"/"+verf_actions[i])

##########################################
card_path = 'Data/Document_Aliveness_selected_card/'+st.session_state['user_id']
if 'Document Aliveness Verification' in st.session_state['selected_verifications']:
    st.write("________")

    st.header('Document Aliveness')
    st.subheader("Selected document")
    doc = os.listdir('Data/Document_Aliveness_selected_card/'+st.session_state['user_id'])
    st.image('Data/Document_Aliveness_selected_card/'+st.session_state['user_id']+"/"+doc[0])
    st.subheader("Verified Actions")
    verf_actions = os.listdir('Data/Document_Aliveness_Verified_Actions/'+st.session_state['user_id'])
    for i in range(len(verf_actions)):
        st.write(verf_actions[i].replace(".jpg",""))
        st.image('Data/Document_Aliveness_Verified_Actions/'+st.session_state['user_id']+"/"+verf_actions[i])

if st.button(label="Back",key="back13"):
    switch_page_button.switch_page(st.session_state['selected_verifications'][-1])

if st.button(label="Select another verification methods",key="back14"):
    switch_page_button.switch_page("select verification methods")