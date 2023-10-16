import streamlit as st
from streamlit_extras import switch_page_button
import time ,uuid
import os
st.set_page_config(
    page_title="Verification",initial_sidebar_state="collapsed"
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

verf_pages = ['Face Verification','Active Aliveness Verification','Document Aliveness Verification','Audio Verification']
# st.sidebar.success("Select a demo above.")
st.subheader("Selecting Verification Methods")
st.write("here we are selecting verification methods ( one or more )")
st.write("____")
if "detect_class_process_results" in st.session_state:
    if st.session_state['detect_class_process_results'] != {}:
        face_verify = st.checkbox('Face Verification')
        st.info("verification between the previously processed image and captured image from camera")
        active_verify = st.checkbox('Active Aliveness Verification') #if face_verify else st.checkbox('Active Aliveness Verification',value=False,disabled=True)
        st.info("verification between the previously processed image and live stream of person in the processed document")
        document_verify = st.checkbox('Document Aliveness Verification') #if face_verify else st.checkbox('Document Aliveness Verification',value=False,disabled=True)
        st.info("verification between the previously processed image and live stream of card")
        audio_verify = st.checkbox('Audio Verification')
        st.info("verification through audio sample")
        st.session_state['verifications'] = [face_verify,active_verify,document_verify,audio_verify]
        st.session_state['selected_verifications'] = [ page for page in verf_pages if st.session_state['verifications'][verf_pages.index(page)] ]
        col1, col2  = st.columns([0.85,0.15],gap="large")
        with col1:
            if st.button(label="Back",key="back4"):
                switch_page_button.switch_page("detect class process")
        with col2:
            if st.button(label="Next",key="next3"):
                switch_page_button.switch_page(st.session_state['selected_verifications'][0])#"Face Verification")
else:
    st.warning("please back to the detect_class_process page then upload image and process it to extract data")
    if st.button(label="Back",key='back3'):
        switch_page_button.switch_page("detect class process")