import json
import sys
import os
from os.path import dirname, abspath
import shutil
from tqdm import tqdm
root = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root)
from configs.configure import uag_oops_video_dir 
from src.data_loaders.uag_oops_loader import UagOopsV1_DataLoader
from src.model_loaders.video_llama2_loader import VideoLlama2Loader
from src.text_representation_builders.utils import save_temporary_frames_from_video
def generate_text_representation_from_video(videollama2, video_path=""):
  temp_video_frames_dir = "./videollama_temp_frames"
  save_temporary_frames_from_video(video_path,output_path=temp_video_frames_dir)
  
  video_frames = [os.path.join(temp_video_frames_dir ,f) for f in os.listdir(temp_video_frames_dir) if f.endswith('.png')]
  sorted_video_frames = sorted(video_frames,key = lambda x: int(x.split("/")[-1].split(".")[0]))
  text_input = "What is happeing in the image? Instruction: answer within one line and cover all the details"
  text_rep = ""
  for i,frame in enumerate(sorted_video_frames):
    text_rep += f"{i}.0s: "
    text_rep += videollama2.infer(video_path=None, gr_img = frame,text_input=text_input)+".\n"
    print(text_rep)
    # delete temp frames inside the folder
  shutil.rmtree(temp_video_frames_dir)
  print(f"text rep : {text_rep}")
  return text_rep
def build_videollama2_text_rep_uag_oops(uag_oops,videollama2,output_path =""):
  videollama2_text_rep_x_uag_oops = {}
  if os.path.exists(output_path):
    with open(output_path) as f:
      videollama2_text_rep_x_uag_oops = json.load(f)
      print(f"size of videollama2_text_rep_x_uag_oops: {len(videollama2_text_rep_x_uag_oops)}")
  saved_rep ={}
  for video_id,video_info in videollama2_text_rep_x_uag_oops.items():
    saved_rep[video_id[:-5]] = video_info["text_rep"]
  for video_id,video_info in tqdm(uag_oops):
    if video_id not in videollama2_text_rep_x_uag_oops:
      print(f"video_id: {video_id}")
      video_path = video_info["video_path"]

      text_rep = generate_text_representation_from_video(videollama2,video_path)
      video_info["text_rep"] = text_rep
      videollama2_text_rep_x_uag_oops[video_id] = video_info
      with open(output_path,'w') as f:
        json.dump(videollama2_text_rep_x_uag_oops,f,indent = 4)
        print(f"successfully saved videollama2_text_rep_x_uag_oops with {len(videollama2_text_rep_x_uag_oops)} samples")
  with open(output_path) as f:
    videollama2_text_rep_x_ssbd_dataset = json.load(f)
    print(f"size of videollama2_text_rep_x_ssbd_dataset: {len(videollama2_text_rep_x_ssbd_dataset)}")

if __name__ == "__main__":
  uag_oops= UagOopsV1_DataLoader()
  videollama2 = VideoLlama2Loader()

  
  build_videollama2_text_rep_uag_oops(uag_oops,videollama2,output_path = videollama2.args.output)
