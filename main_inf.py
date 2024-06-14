import numpy as np
import av
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import streamlit as st
import torch.nn as nn


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# def sample_frame_indices2(clip_len, frame_sample_rate, seg_len):
#     '''
#     Description
#     Args:
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     '''
#     return 



def classify(file):
    container = av.open(file)

    # sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    # st.write(f'Les labels: {model.config.id2label}')
    # st.write(f'répartiton des probilités {logits}')
    # st.write(f'répartiton des probilités {nn.Softmax(dim=-1)(logits)}')
    
    return model.config.id2label[predicted_label]


model_ckpt = '2nzi/videomae-surf-analytics'
# pipe = pipeline("video-classification", model="2nzi/videomae-surf-analytics")
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt)





st.subheader("Surf Analytics")

st.markdown("""
    Bienvenue sur le projet Surf Analytics réalisé par Walid, Guillaume, Valentine, et Antoine.
            
    <a href="https://github.com/2nzi/M09-FinalProject-Surf-Analytics" style="text-decoration: none;">@Surf-Analytics-Github</a>.
""", unsafe_allow_html=True)

st.title("Surf Maneuver Classification")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    st.video(video_bytes)
    predicted_label = classify(uploaded_file)
    st.success(f"Predicted Label: {predicted_label}")