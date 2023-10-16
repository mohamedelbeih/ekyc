import streamlit as st
from streamlit_extras import switch_page_button

st.set_page_config(
    page_title="Audio Verification",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

# st.sidebar.success("Select a demo above.")
st.subheader("Audio Verification")
st.write("here Verification with audio sample")
st.write("____")

page_name = 'Audio Verification'
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
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<",st.session_state['user_id'])
if page_name in st.session_state['selected_verifications']:
    st.write("inprogress")
    col1, col2  = st.columns([0.85,0.15],gap="large")
    with col1:
        if st.button(label="Back",key="back11"):
            switch_page_button.switch_page(previous_page)
    with col2:
        if st.button(label="Next",key="next7"):
            switch_page_button.switch_page(next_page)

else:
    st.warning("please back to the select verification methods page then select active aliveness verification ")
    if st.button(label="Back",key='back12'):
        switch_page_button.switch_page("select verification methods")