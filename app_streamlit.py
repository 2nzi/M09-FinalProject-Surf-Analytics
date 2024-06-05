#docker run -it -v "$(pwd):/home/app" -p 4000:4000 jedha/streamlit-fs-image
#docker run -it -v "$(pwd):/home/app" -p 4000:4000 jedha/streamlit-fs-image bash

#http://localhost:4000

import streamlit as st
import requests




st.subheader("Surf Analytics")

st.markdown("""
    Bienvenue sur le projet Surf Analytics` réalisé par Walid Guillaume Valentine et Antoine.
    <a href="https://twitter.com" style="text-decoration: none;">@createur_link</a>.
""", unsafe_allow_html=True)


st.title("Surf Maneuver Classification")


uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the video file
    video_bytes = uploaded_file.read()

    # Display the video
    st.video(video_bytes)

    files = {'file': uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:4000/classify", files=files)

    if response.status_code == 200:
        st.success(f"Predicted Label: {response.json()}")
    else:
        st.error("Error in prediction")




