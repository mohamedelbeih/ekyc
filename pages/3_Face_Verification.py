import streamlit as st
from streamlit_extras import switch_page_button
import base64
import json
import os

st.set_page_config(
    page_title="Face Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.subheader("Face verification through camera")
st.write("here verifying detected face in the previous given image with a live image from camera ( real person or another image )")
st.write("____")

page_name = 'Face Verification'
try:
    page_idx_in_selected = st.session_state['selected_verifications'].index(page_name)
    if len(st.session_state['selected_verifications']) > 1:
        next_page =  st.session_state['selected_verifications'][page_idx_in_selected+1]
    else:
        next_page = "end of verifications"
except:
    pass
#verf_pages = ['Face Verification','Active Aliveness Verification','Document Aliveness Verification','Audio Verification']

if page_name in st.session_state['selected_verifications']:
    uploaded_picture = st.file_uploader("upload Image", type=["png","jpg","jpeg", 'bmp', 'tiff'], key='face_imge')
    captured_picture = st.camera_input("Or Take a picture")
    if captured_picture or uploaded_picture:
        picture = captured_picture if captured_picture else uploaded_picture
        st.write("Captured image")
        st.image(picture)
        img_bytes = picture.getvalue()
        with open("verify/Input/sample/captured_img.jpg","wb") as f:
            f.write(img_bytes)

    st.write("previously given image :")
    img= base64.b64decode(st.session_state['image'].encode('utf-8'))
    st.image(img)
    with open("verify/Input/sample/processed_img.jpg","wb") as f:
        f.write(img)
        
    #verifying
    not_matched = True
    if st.button("verify"):
        os.system("cd verify; export LD_LIBRARY_PATH=$(pwd);python rdi_face_verification_test.py -I Input/ -O Output/ -M face_model/model_enc/")
        with open("verify/Output/sample/metric.txt") as f:
            lines = f.readlines()
            matching_percent =  lines[-1].replace("confidence=","").strip()
        
        # matching_percent = 0.6
        if float(matching_percent) > 50:
            st.success("Percent of matching = {}".format(matching_percent))
            not_matched = False
        else:
            st.warning("Percent of matching = {}".format(matching_percent))
            not_matched = True
            if st.button("Try again"):
                switch_page_button.switch_page(page_name)

    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back6"):
            switch_page_button.switch_page("select verification methods")
    with col2:
        if st.button(label="Next",key="next4",disabled=not_matched):
            switch_page_button.switch_page(next_page)
# elif st.session_state['verifications'][1]:
#     switch_page_button.switch_page("Active Aliveness Verification")
else:
    st.warning("please back to the select verification methods page then select face verification")
    if st.button(label="Back",key='back5'):
        switch_page_button.switch_page("select verification methods")
