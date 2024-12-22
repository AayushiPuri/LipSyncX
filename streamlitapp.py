# Import all of the dependencies
import streamlit as st
import os 
from mouth_roi_util import preprocess_video
from pipeline_util import InferencePipeline

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="LipSyncX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Setup the sidebar
with st.sidebar: 
    st.image('assets//transformer_image.jpeg')
    st.title('LipSyncX')
    st.info('This application is developed using the LipSyncX deep learning model.')

st.title('LipSyncX App') 


# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'samples'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('samples', selected_video)
        print(file_path)
        video = open(file_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
 

    with col2: 
        st.info('Mouth ROI Seen By The Model')
        preprocess_video(src_filename=file_path, dst_filename='test_video_roi.mp4')
        video = open('test_video_roi.mp4', 'rb') 
        video_bytes = video.read()
        st.video(video_bytes)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        create_embeddings = st.checkbox("✅ Predict")
        if create_embeddings:
            modality = "video"
            model_conf = "LRS3_Model//model.json"
            model_path = "LRS3_Model//model.pth"
            print(model_conf, model_path)
            pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)
            transcript = pipeline(file_path)
            st.text(transcript)


            st.info('This is the output of the machine learning model')
            features = pipeline.extract_features(file_path)
            st.text(features)
            st.text(features.size())