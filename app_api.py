import numpy as np
import av
import torch
from huggingface_hub import hf_hub_download
from transformers import VideoMAEForVideoClassification, pipeline, AutoImageProcessor, AutoModelForVideoClassification
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile


model_ckpt = '2nzi/videomae-surf-analytics'
pipe = pipeline("video-classification", model="2nzi/videomae-surf-analytics")
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt)


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



description = """
Description à faire
"""

app = FastAPI(
    title="Surf Analytics",
    description=description,
    version="0.1",
    contact={
        "name": "Guillaume Valentine Walid Antoine",
        "url": "antoineverdon.pro@gmail.com",
    }
)


app = FastAPI()


@app.post("/post-picture", tags=["Blog Endpoints"])
async def post_picture(file: UploadFile= File(...)):
    """
    Upload a picture and read its file name.
    """
    return {"picture": file.filename}


class PredictionFeatures(BaseModel):
    SurfManveuver: str ="demo-cutback-frontside.mp4"



@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    """
    Prediction of ...
    """

    # file_path = hf_hub_download(repo_id="2nzi/surf-maneuvers", filename=file.filename, repo_type="dataset")
    print(file.filename)
    container = av.open(file.file)

    # sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(model_ckpt)

    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    
    return model.config.id2label[predicted_label]


# @app.post("/predict")
# async def predict(predictionFeatures: PredictionFeatures):
#     """
#     Prediction of ...
#     """

#     file_path = hf_hub_download(repo_id="2nzi/surf-maneuvers", filename=predictionFeatures.SurfManveuver, repo_type="dataset")
#     print(predictionFeatures.SurfManveuver)
#     container = av.open(file_path)

#     # sample 16 frames
#     indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
#     video = read_video_pyav(container, indices)

#     image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
#     model = VideoMAEForVideoClassification.from_pretrained(model_ckpt)

#     inputs = image_processor(list(video), return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     # model predicts one of the 400 Kinetics-400 classes
#     predicted_label = logits.argmax(-1).item()
#     print(model.config.id2label[predicted_label])
    
#     return model.config.id2label[predicted_label]





if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)




# @app.post("/batch-pred")
# async def batch_pred(file: UploadFile = File(...)):
#     """
#     Make batch predictions 

#     """
#     df = pd.read_excel(file.file)

#     # Log model from mlflow 
#     logged_model = 'runs:/5e54b2ee620546b0914c9e9fbfd18875/salary_estimator'

#     # Load model as a PyFuncModel.
#     loaded_model = mlflow.pyfunc.load_model(logged_model)
#     prediction = loaded_model.predict(df)

#     # Format response
#     response = {"prediction": prediction.tolist()}

#     return response