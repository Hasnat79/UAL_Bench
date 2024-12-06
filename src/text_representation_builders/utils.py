import cv2
import os
import torch
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)

from transformers import Blip2Processor, Blip2ForConditionalGeneration
def load_frame(frame_path):
    '''loads a frame from a given frame path
    '''
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def save_temporary_frames_from_video(video_path="", output_path = "./temp_frames"):
    '''from a video path it saves the first frame and the rest of the frames at 1fps in a temp folder
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    os.system(f"""ffmpeg -i "{video_path}" -vf fps=1 {output_path}/%d.png""")


@memory.cache
def initialize_blip2_model():
    '''
    initializes the blip2 model
    '''
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",cache_dir = cache_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
"Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir = cache_dir) 
    model.to("cuda:0")
    return processor,model
