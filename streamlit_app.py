import streamlit as st
from streamlit_extras import switch_page_button
import os , uuid
st.set_page_config(
    page_title="Home",initial_sidebar_state="collapsed"
    # page_icon="👋",
)

st.write("# Welcome to E-KYC 👋")
st.markdown(
    """
    **E-KYC project** provides a pipeline to do the following:
    - building a customized document classifier of our supported documents
    - processing the documents in a given image , extracting data
    - verifying the person in the processed document through live camera ( from another document or the real person)
    - Document aliveness verification
    - User aliveness verification
    - Audio aliveness verification
    ### below .. you will need to check the wanted documents:
"""
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

st.session_state['user_id'] = str(uuid.uuid4())

check_all = st.checkbox("Select all")
st.write("")
document0 = st.checkbox('SaudiArabia_dl',value=check_all)
document1 = st.checkbox('Pakistan_ids',value=check_all)
document2 = st.checkbox('Kuwait_dl',value=check_all)
document3 = st.checkbox('saudi_card_v1_ids',value=check_all)
document4 = st.checkbox('algeria_ids',value=check_all)
document5 = st.checkbox('AmericaTexas_dl',value=check_all)
document6 = st.checkbox('Morocco_ids',value=check_all)
document7 = st.checkbox('iraq_ids',value=check_all)
document8 = st.checkbox('bahrain_ids',value=check_all)
document9 = st.checkbox('ahly_bank_receipts',value=check_all)
document10 = st.checkbox('AmericaNewYork_dl',value=check_all)
document11 = st.checkbox('somali_ids',value=check_all)
document12 = st.checkbox('yemen_ids',value=check_all)
document13 = st.checkbox('cib_bank_receipts',value=check_all)
document14 = st.checkbox('saudi_card_resident_permit',value=check_all)
document15 = st.checkbox('Egypt_Residence-Permits',value=check_all)
document16 = st.checkbox('Egyptian_id_back',value=check_all)
document17 = st.checkbox('Egyptian_vehicle_license_front',value=check_all)
document18 = st.checkbox('qatar_resident',value=check_all)
document19 = st.checkbox('tunisia_ids',value=check_all)
document20 = st.checkbox('saudi_family_residence_permit',value=check_all)
document21 = st.checkbox('qnb_bank_receipts',value=check_all)
document22 = st.checkbox('kuwait_ids_v2',value=check_all)
document23 = st.checkbox('Indiadl_v2',value=check_all)
document24 = st.checkbox('Qatar_dl',value=check_all)
document25 = st.checkbox('djibuti_ids',value=check_all)
document26 = st.checkbox('oman_ids',value=check_all)
document27 = st.checkbox('comoros_ids_v1',value=check_all)
document28 = st.checkbox('Masr_bank_receipts',value=check_all)
document29 = st.checkbox('Algeria_dl',value=check_all)
document30 = st.checkbox('qatar_ids',value=check_all)
document31 = st.checkbox('Egyptian_ids_front',value=check_all)
document32 = st.checkbox('Egyptian_driving_license',value=check_all)
document33 = st.checkbox('kuwait_ids',value=check_all)
document34 = st.checkbox('Kenya_Ids',value=check_all)
document35 = st.checkbox('India_dl_v3',value=check_all)
document36 = st.checkbox('Iraq1_dl',value=check_all)
document37 = st.checkbox('UAE_dl',value=check_all)
document38 = st.checkbox('UAE_ids',value=check_all)
document39 = st.checkbox('SouthAfrica_ids',value=check_all)
document40 = st.checkbox('sudan_ids',value=check_all)
document41 = st.checkbox('syria_ids',value=check_all)
document42 = st.checkbox('Oman_dl',value=check_all)
document43 = st.checkbox('Ghana_dl',value=check_all)
document44 = st.checkbox('Morocco_dl',value=check_all)
document45 = st.checkbox('saudi_ids_v2',value=check_all)
document46 = st.checkbox('India_dl_v1',value=check_all)
document47 = st.checkbox('Tunisia_dl',value=check_all)
document48 = st.checkbox('Egyptian_vehicle_license_back',value=check_all)
document49 = st.checkbox('Saudi_Gold-Residence-Permits',value=check_all)
document50 = st.checkbox('mauritania_ids',value=check_all)
document51 = st.checkbox('comoros_ids_v2',value=check_all)
document52 = st.checkbox('jordan_ids',value=check_all)
document53 = st.checkbox('kuwait_ids_v1',value=check_all)
document54 = st.checkbox('lebanon_ids',value=check_all)
document55 = st.checkbox('Nigeria_ids',value=check_all)
document56 = st.checkbox('Mrz',value=check_all)

st.session_state['documents'] = [document0, document1, document2, document3, document4, document5,
                                  document6, document7, document8, document9, document10, document11,
                                  document12, document13, document14, document15, document16, document17,
                                  document18, document19, document20, document21, document22, document23,
                                  document24, document25, document26, document27, document28, document29,
                                  document30, document31, document32, document33, document34, document35,
                                  document36, document37, document38, document39, document40, document41,
                                  document42, document43, document44, document45, document46, document47,
                                  document48, document49, document50, document51, document52, document53,
                                  document54, document55,document56]

if st.button(label="Next",key="next1"):
    switch_page_button.switch_page("detect class process")  