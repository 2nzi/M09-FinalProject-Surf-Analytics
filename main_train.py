#%%

#----------------------------------------------------------------------------------------------------------
#------------------------------------------------ IMPORT LIB ----------------------------------------------
#----------------------------------------------------------------------------------------------------------

import torch
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import pytorchvideo.data
import os
import pathlib
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize, ApplyTransformToKey
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------
#------------------------------------------------ CREATE FUNCTIONS ----------------------------------------
#----------------------------------------------------------------------------------------------------------

def save_confusion_matrix(labels, predictions, label_names, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(labels, predictions, labels=np.arange(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.xticks(rotation=45)
    ax.set_title(title)
    plt.savefig(save_path)
    plt.close()

accuracy_metric = load_metric('accuracy')
f1_metric = load_metric('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


# Define Collate Function
def collate_fn(examples):
    pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


#----------------------------------------------------------------------------------------------------------
#--------------------------------------------- IMPORT DATA PATH -------------------------------------------
#----------------------------------------------------------------------------------------------------------


dataset_root_path = 'data-split'
dataset_root_path = pathlib.Path(dataset_root_path)
train_test_val_dataset_path = [item.name for item in dataset_root_path.glob("**") if item.is_dir()]


# Iterate over surf classes and aggregate mp4 files
all_video_file_paths = []
for surf_class in train_test_val_dataset_path:
    surf_class_path = dataset_root_path / surf_class
    mp4_files = surf_class_path.glob("**/*.mp4")
    all_video_file_paths.extend(mp4_files)

all_video_file_paths = list(all_video_file_paths)

class_labels = sorted({str(path).split("\\")[-2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")


#----------------------------------------------------------------------------------------------------------
#------------------------------------------ CHOOSE MODEL TO FINE TUNE -------------------------------------
#----------------------------------------------------------------------------------------------------------

model_id = 1

# VIDEOMAE
if model_id==1:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    video_processor = "MCG-NJU/videomae-base"  # image processing from pre-trained model
    model_ckpt = video_processor  # pre-trained model from which to fine-tune
    video_processor = VideoMAEImageProcessor.from_pretrained(video_processor)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

# VIVIT
if model_id==2:
    from transformers import VivitForVideoClassification,VivitImageProcessor

    video_processor = "google/vivit-b-16x2-kinetics400"  
    model_ckpt = "google/vivit-b-16x2-kinetics400"
    video_processor = VivitImageProcessor.from_pretrained(video_processor)
    model = VivitForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )


#TIMESFORMER
if model_id==3:
    from transformers import VideoMAEImageProcessor,TimesformerForVideoClassification
    video_processor = "MCG-NJU/videomae-base"
    model_ckpt = "facebook/timesformer-base-finetuned-k400" 

    video_processor = VideoMAEImageProcessor.from_pretrained(video_processor)
    model = TimesformerForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

#----------------------------------------------------------------------------------------------------------
#-------------------------- --------------- VIDEO TRANSFORMATION ------------------------------------------
#----------------------------------------------------------------------------------------------------------


mean = video_processor.image_mean
std = video_processor.image_std
if "shortest_edge" in video_processor.size:
    height = width = video_processor.size["shortest_edge"]
else:
    height = video_processor.size["height"]
    width = video_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((224, 224)),

                ]
            ),
        ),
    ]
)


#----------------------------------------------------------------------------------------------------------
#------------------------------------------ APPLY TRANFORMATION -------------------------------------------
#----------------------------------------------------------------------------------------------------------


# Training dataset.
train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=transform,
)

# Validation and evaluation datasets.
val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=transform,
)


#----------------------------------------------------------------------------------------------------------
#--------------------------------------------- TRAIN NEW MODEL --------------------------------------------
#----------------------------------------------------------------------------------------------------------


new_model_name = "videomae-surf-analytics-sans-wandb"
# new_model_name = f"videomae-surf-analytics-{model}"
num_epochs = 5
batch_size = 2

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    # metric_for_best_model="accuracy",
    push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    report_to="wandb"
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=video_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


#----------------------------------------------------------------------------------------------------------
#------------------------------------------------ SAVE METRICS ---------------------------------------------
#----------------------------------------------------------------------------------------------------------

print('Start Train')
train_results = trainer.train()
print('End Train')

print('Start Evaluate')
train_evaluate = trainer.evaluate(train_dataset)
test_evaluate = trainer.evaluate(test_dataset)
val_evaluate = trainer.evaluate(val_dataset)
print('End Evaluate')

print('Start log metrics and model')
trainer.log_metrics("test", test_evaluate)
trainer.save_metrics("test", test_evaluate)
trainer.log_metrics("val", val_evaluate)
trainer.save_metrics("val", val_evaluate)
trainer.save_state()
trainer.save_model()

test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_labels = test_predictions.label_ids

new_model_name = "confusion_matrix.png"
save_confusion_matrix(test_labels, test_preds, list(label2id.keys()), './' + new_model_name, title='My Model Confusion Matrix')
print('End log metrics and model')
print('End')


# %%
