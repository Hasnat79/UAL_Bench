import json
import sys
import argparse
import os
from os.path import dirname, abspath
import shutil
from tqdm import tqdm
root = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root)
from configs.configure import  charades_video_dir
from src.data_loaders.charades_loader import Charades_STA_Dataloader
from src.model_loaders.video_llama2_loader import VideoLlama2Loader
from src.text_representation_builders.utils import save_temporary_frames_from_video


def generate_text_representation_from_video(videollama2, video_path=""):
  temp_video_frames_dir = "./videollama_temp_frames"
  save_temporary_frames_from_video(video_path,output_path=temp_video_frames_dir)
  
  video_frames = [os.path.join("./videollama_temp_frames" ,f) for f in os.listdir("./videollama_temp_frames") if f.endswith('.png')]
  sorted_video_frames = sorted(video_frames,key = lambda x: int(x.split("/")[-1].split(".")[0]))
  text_input = "What is happening in the image? Instruction: answer within one line and cover all the details"
  text_rep = ""
  for i,frame in enumerate(sorted_video_frames):
    text_rep += f"{i}.0s: "
    text_rep += videollama2.infer(video_path=None, gr_img = frame,text_input=text_input)+"\n"
    print(text_rep)
    # delete temp frames inside the folder
  shutil.rmtree("./videollama_temp_frames")
  print(f"text rep : {text_rep}")
  return text_rep

def build_videollama2_text_rep(data = '',model = "",output_path =""):
  videollama2_text_rep_x_charades_sta = []
  text_rep_buffer = {}
  if os.path.exists(output_path):
    with open(output_path,'r') as f:
      videollama2_text_rep_x_charades_sta = json.load(f)
      print(f"videollama2_text_rep_x_charades_sta loaded with {len(videollama2_text_rep_x_charades_sta)} samples")
  start =0

  if len(videollama2_text_rep_x_charades_sta)>0:
    start = len(videollama2_text_rep_x_charades_sta)

  for sample in tqdm(data):
    video_id = sample['video_id']
    if video_id not in videollama2_text_rep_x_charades_sta:
      video_path = video_path = os.path.join(root,charades_video_dir, video_id)+".mp4"
      print(f"video_path: {video_path}")
      if video_id in text_rep_buffer:
        sample["text_rep"] = text_rep_buffer[video_id]
        # videollama2_text_rep_x_charades_sta[video_id] = sample
        videollama2_text_rep_x_charades_sta.append(sample)
        with open(output_path, 'w') as f:
          json.dump( videollama2_text_rep_x_charades_sta, f,indent=4)
          print(f"succesfully saved {output_path} with {len( videollama2_text_rep_x_charades_sta)} samples")
        continue
      text_rep = generate_text_representation_from_video(model,video_path)
      text_rep_buffer[video_id] = text_rep
      sample["text_rep"] = text_rep
      # videollama2_text_rep_x_funqa[video_id] = sample
      videollama2_text_rep_x_charades_sta.append(sample)
      with open(output_path, 'w') as f:
        json.dump( videollama2_text_rep_x_charades_sta, f,indent=4)
        print(f"succesfully saved{output_path} with {len( videollama2_text_rep_x_charades_sta)} samples")
    break
  #check count
  with open(output_path) as f:
    videollama2_text_rep_x_charades_sta = json.load(f)
    print(f"size of {output_path}: {len(videollama2_text_rep_x_charades_sta)}")


if __name__ == "__main__":
  videollama2 = VideoLlama2Loader()
  charades_sta_dataset = Charades_STA_Dataloader()
  
  # with open(charades_sta_test_path, 'r') as f:
  #   charades_sta_dataset = json.load(f)
  print(f"charades sta test-set loaded with {len(charades_sta_dataset)} videos")
  build_videollama2_text_rep(charades_sta_dataset,videollama2,output_path = videollama2.args.output)
