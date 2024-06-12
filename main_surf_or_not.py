#%%
import numpy as np
import av
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch.nn as nn

# Function to read video frames using PyAV
def read_video_pyav(container, indices):
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

# Function to sample frame indices from a video
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Function to classify the sport in a video
def classify(file):
    container = av.open(file)
    # Sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    
    # Preprocess video frames
    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()
    print(f'Les labels: {model.config.id2label}')
    print(f'répartiton des probabilités {logits}')
    print(f'répartiton des probabilités {nn.Softmax(dim=-1)(logits)}')

    return model.config.id2label[predicted_label]

# Load the model and image processor
model_ckpt = '2nzi/videomae-surf-analytics-surfNOTsurf'
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt)
#%%


# Example usage
choose = False
if choose:
    # surf_video_path
    video_path = r"C:\Users\antoi\Documents\Work_Learn\JEDHA\M09-FinalProject-Surf-Analytics\data-split\test\360\360.mp4"
else:
    # non_surf_video_path
    video_path= r"C:\Users\antoi\Documents\Work_Learn\JEDHA\M09-FinalProject-Surf-Analytics\Doc\test.mp4"


predicted_label = classify(video_path)
print(f"Predicted Label: {predicted_label}")

# %%
