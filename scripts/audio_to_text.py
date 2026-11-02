#!/usr/bin/env python
# coding: utf-8

# In[32]:

from pathlib import Path
import json
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd




# In[30]:


device = "cuda:0"
dtype = torch.float16

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)


# In[3]:


# result = pipe(
#     "./Semantic Grounding Project/audio/segments/1-1_seg_34.30-34.70.wav",
#     generate_kwargs={
#         "task": "transcribe", 
#         "language": "en"
#     }
# )
# print(result["text"])


# In[4]:


root_dir = Path("/home/simbanyatsanga/Projects/phd_work/research/experiments/semantic-clustering/data/Trinity_Dataset_2_v2/audio/segments")


# In[5]:


audio_extensions = {".wav"}


# In[6]:


audio_data = []
audio_suffix_set = set()



for audio_path in root_dir.rglob("*"):


    if audio_path.suffix.lower() in audio_extensions:

        # Assuming your structure is root_dir / category_folder / video.mp4
        # The parent folder name acts as the category/label
        category = audio_path.parent.name
        filename = audio_path.name
        relative_path = str(audio_path.relative_to(root_dir))
        absolute_path = str(audio_path.resolve())

        audio_data.append({
            "filename": filename,
            "category": category,
            "relative_path": relative_path,
            "absolute_path": absolute_path,
            "file_size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2),
        })
#     else:
#         audio_suffix_set.add(audio_path)
#         print(f"{audio_path} extension not processed")

# print(audio_suffix_set)


# In[7]:


audio_df = pd.DataFrame(audio_data)


# In[8]:


print(f"Total audio files found: {len(audio_df)}")
# display(audio_df.head())

# find . -type f | wc -l # same result


# In[34]:


file_to_transcription = {}

#list of all paths needed
absolute_paths = [row['absolute_path'] for index, row in audio_df.iterrows()]
relative_paths = [row['relative_path'] for index, row in audio_df.iterrows()]

start_time = time.perf_counter()
results = pipe(
    absolute_paths, 
    batch_size=8, 
    return_timestamps=True,
    generate_kwargs={
        "task": "transcribe", 
        "language": "en",
    }
)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Finished in {elapsed_time:.2f} seconds")
print(f"Average time of {elapsed_time / len(absolute_paths):.4f} seconds per file")

for rel_path, res in zip(relative_paths, results):
    file_to_transcription[rel_path] = res["text"]


# In[26]:


print(f"{len(file_to_transcription)} transcriptions completed")


# In[28]:


with open("file_to_transcription.json", "w") as f:
    json.dump(file_to_transcription, f)

#Result
# Finished in 793.65 seconds
# Average time of 0.2217 seconds per file
# potential error of 2–3% word error rate (WER) on clear audio or 7.4% WER on most english language benchmarks https://loronote.com/en/blog/how-accurate-is-whisper




