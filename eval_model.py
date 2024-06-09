#%%
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
from torchvision.io import read_video
import pathlib
import pandas as pd
import numpy as np
import av
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import pathlib
from tqdm import tqdm


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
    if seg_len < converted_len:
        raise ValueError("seg_len must be greater than or equal to converted_len")
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# Fonction pour lire les vidéos et les préparer pour le modèle
def process_video(video_path, image_processor):
    video, _, _ = read_video(video_path)
    video = video.permute(0, 3, 1, 2)  # Reorder dimensions for PyTorch (T, C, H, W)
    inputs = image_processor(videos=video.unsqueeze(0), return_tensors="pt")
    return inputs

# Calculer le F1-score et d'autres métriques
def evaluate(predictions, true_labels):
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return f1, accuracy, report


def get_mp4_files(dataset_root_path: str):
    """
    Retrieve all mp4 files from the given dataset root path, organized by surf classes.

    Args:
    dataset_root_path (str): The root directory containing subdirectories for each surf class.

    Returns:
    List[pathlib.Path]: A list of paths to all mp4 files found in the dataset.
    """
    dataset_root_path = pathlib.Path(dataset_root_path)
    surf_classes = [item.name for item in dataset_root_path.glob("*") if item.is_dir()]

    # List to store all mp4 files
    full_dataset = []

    # Iterate over surf classes and aggregate mp4 files
    for surf_class in surf_classes:
        surf_class_path = dataset_root_path / surf_class
        mp4_files = surf_class_path.glob("**/*.mp4")
        full_dataset.extend(mp4_files)

    return list(full_dataset)


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
    # print(model.config.id2label[predicted_label])
    
    return model.config.id2label[predicted_label]


# Charger le modèle et le processeur d'image
model_ckpt = '2nzi/videomae-surf-analytics'
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt)
model.eval()

dataset_root_path = 'data-split'
files = get_mp4_files(dataset_root_path)

files_data = {
    'file_path': [file for file in files],
    'file_parent': [file.parent.name for file in files],
    'file_parent_parent': [file.parent.parent.name for file in files]
}

files_df = pd.DataFrame(files_data)
predicted_label_list=[]

for file in tqdm(files_df['file_path']):
    try:
        real_label = file.split('/')[-2]
        predicted_label = classify(file)
        predicted_label_list.append(predicted_label)
        print(real_label, predicted_label)
    except ValueError as e:
        print(f"Error processing {file}: {e}")
        predicted_label_list.append(str(e))


#%%
# Afficher le DataFrame
predicted_labels_df = pd.DataFrame(predicted_label_list, columns=['predicted_label'])
files_df = pd.concat([files_df, predicted_labels_df], axis=1)


#%%
# files_df.to_csv('test.csv', index=False)

# %%


files_df = pd.read_csv('label_real_pred.csv')
files_df[files_df['file_parent']==files_df['predicted_label']].count()
files_df[files_df['file_parent']!=files_df['predicted_label']].count()

files_df['predicted_label'].value_counts()


# files_df = files_df[files_df['file_parent_parent']=='test']
# files_df = files_df[files_df['file_parent_parent']=='val']
# files_df = files_df[files_df['file_parent_parent']=='train']

true_labels,pred_labels=files_df['file_parent'],files_df['predicted_label']
accuracy_score(true_labels, pred_labels)
f1_score(true_labels, pred_labels,average='weighted')


import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



labels = true_labels.unique()
labels = pred_labels.unique()
true_labels
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)

# Creating a DataFrame from the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Plotting using matplotlib and seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="viridis")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# %%
