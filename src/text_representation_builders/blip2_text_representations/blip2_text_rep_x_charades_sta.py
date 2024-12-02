import os
from os.path import dirname, abspath
import sys
import shutil
import cv2
import torch

from tqdm import tqdm
root = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root)

from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)


import json
from configs.configure import charades_sta_test_path, charades_video_dir,funqa_data_path, funqa_test_humor_video_dir
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse


@memory.cache
def initialize_blip2_model():
    '''
    initializes the blip2 model
    '''
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",cache_dir = cache_dir,revision="51572668da0eb669e01a189dc22abe6088589a24")
    model = Blip2ForConditionalGeneration.from_pretrained(
"Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir = cache_dir,revision="51572668da0eb669e01a189dc22abe6088589a24") 
    model.to("cuda:0")
    return processor,model


def vqa_captioner(frame,question=""):
    '''
    given a frame / image it returns the captioned text using blip2 model utilizing visual question answering technique
    '''
    # question = "Question: What is happening in the image? Answer: "
    inputs = processor(frame, return_tensors="pt").to("cuda",torch.float16)
    out = model.generate(
    **inputs,
    num_beams=5,
    min_length=24,
    max_length=72
)   
    generated_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    
    print(f"Generated text: {generated_text}\n\n")

    return generated_text


def save_temporary_frames_from_video(video_path="", output_path = "./temp_frames"):
    '''from a video path it saves the first frame and the rest of the frames at 1fps in a temp folder
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    # extract frames from video: 1fps -- 0th second ~ frame 1

    os.system(f"""ffmpeg -i "{video_path}" -vf fps=1 {output_path}/%d.png""")
    # save frames in output_path
    
def load_frame(frame_path):
    '''loads a frame from a given frame path
    '''
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
def generate_text_representation_from_video(video_path=""):
    ''' from a video path it saves all the frames at 1fp in a temp folder and then generates text representation from each frame using captioner function and returns the concatenated text representation of all the frames.
    '''
    temp_video_frames_dir = "./temp_frames"
    save_temporary_frames_from_video(video_path,output_path = temp_video_frames_dir)
    video_frames = [os.path.join(temp_video_frames_dir, f) for f in os.listdir(temp_video_frames_dir) if f.endswith('.png')]
    sorted_video_frames = sorted(video_frames,key = lambda x: int(x.split("/")[-1].split(".")[0]))
    text_rep = ""
    
    for i,frame in enumerate(sorted_video_frames):
        print(f"frame: {frame}")
        frame_image = load_frame(frame)
        text_rep += f"{i}.0s: "
        text_rep += vqa_captioner(frame_image)+".\n"
        
    # delete temp frames inside the folder
    shutil.rmtree("./temp_frames")

    return text_rep
def build_blip2_text_rep(dataset = "dataset_path",video_dir="video_dir", output_path = "output_path"):
    blip2_text_rep_x_charades_sta = []
    text_rep_buffer = {}
    if os.path.exists(output_path):
        with open(output_path,'r') as f:
            blip2_text_rep_x_charades_sta = json.load(f)
        print(f"Loaded {output_path} with {len(blip2_text_rep_x_charades_sta)} samples")
    start = 0
    if len(blip2_text_rep_x_charades_sta)>0:
        start = len(blip2_text_rep_x_charades_sta)
    print(f"Starting from {start}")
    print(f"dataset[start]: {dataset[start]}")
    for sample in tqdm(dataset[start:]):
        video_id = sample['video_id']
        if video_id not in blip2_text_rep_x_charades_sta:
            video_path = os.path.join(root,video_dir, video_id)+".mp4"
            print(f"video_path: {video_path}")
            if video_id in text_rep_buffer:
                sample["text_rep"] = text_rep_buffer[video_id]
                # blip2_text_rep_x_charades_sta[video_id] = sample
                blip2_text_rep_x_charades_sta.append(sample)
                with open(output_path, 'w') as f:
                    json.dump(blip2_text_rep_x_charades_sta, f,indent=4)
                    print(f"succesfully saved blip2_text_rep_x_charades_sta with {len(blip2_text_rep_x_charades_sta)} samples")
                continue
            text_rep = generate_text_representation_from_video(video_path)
            text_rep_buffer[video_id] = text_rep
            sample["text_rep"] = text_rep
            # blip2_text_rep_x_charades_sta[video_id] = sample
            blip2_text_rep_x_charades_sta.append(sample)

            with open(output_path, 'w') as f:
                json.dump(blip2_text_rep_x_charades_sta, f,indent=4)
                print(f"succesfully saved blip2_text_rep_x_charades_sta with {len(blip2_text_rep_x_charades_sta)} samples")
    #check count
    with open(output_path) as f:
        blip2_text_rep_x_charades_sta = json.load(f)
        print(f"size of blip2_text_rep_x_charades_sta: {len(blip2_text_rep_x_charades_sta)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text representations for videos using BLIP2 model.")
    parser.add_argument('--output', type=str, required=True, help='Output JSON file name')
    args = parser.parse_args()

    with open(os.path.join(root,charades_sta_test_path), 'r') as f:
        charades_sta_dataset = json.load(f)
    print(f"charades sta test-set loaded with {len(charades_sta_dataset)} videos")
    
    processor, model = initialize_blip2_model()

    build_blip2_text_rep(charades_sta_dataset,charades_video_dir, args.output)
