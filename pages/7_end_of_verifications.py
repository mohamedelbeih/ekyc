import streamlit as st
from streamlit_extras import switch_page_button

st.set_page_config(
    page_title="Audio Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.success("Your verifications have been done")

if st.button(label="Back",key="back13"):
    switch_page_button.switch_page(st.session_state['selected_verifications'][-1])

if st.button(label="Select another verification methods",key="back14"):
    switch_page_button.switch_page("select verification methods")