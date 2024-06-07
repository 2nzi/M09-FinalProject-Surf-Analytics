import torch
from transformers import TrainingArguments, Trainer, VideoMAEForVideoClassification, VideoMAEImageProcessor
from datasets import load_metric
import pytorchvideo.data
import os
import pathlib
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize, ApplyTransformToKey


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


class_labels = sorted({str(path).split("\\")[2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")


model_ckpt = "MCG-NJU/videomae-base"  # pre-trained model from which to fine-tune
batch_size = 8  # batch size for training and evaluation

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # Pour le fine tune
)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps




accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=1)
    references = eval_pred.label_ids
    return accuracy_metric.compute(predictions=predictions, references=references)


# def compute_metrics(eval_pred):
#     predictions = eval_pred.predictions.argmax(axis=1)
#     references = eval_pred.label_ids
    
#     accuracy = accuracy_metric.compute(predictions=predictions, references=references)
#     f1 = f1_metric.compute(predictions=predictions, references=references, average='weighted')
    
#     return {
#         "accuracy": accuracy["accuracy"],
#         "f1": f1["f1"]
#     }

# Define Collate Function
def collate_fn(examples):
    pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



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


new_model_name = "videomae-surf-analytics-2"
num_epochs = 1
batch_size = 4

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
train_evaluate = trainer.evaluate(train_dataset)
test_evaluate = trainer.evaluate(test_dataset)
val_evaluate = trainer.evaluate(val_dataset)

trainer.log_metrics("test", test_evaluate)
trainer.save_metrics("test", test_evaluate)
trainer.log_metrics("val", val_evaluate)
trainer.save_metrics("val", val_evaluate)
trainer.save_state()
trainer.save_model()
