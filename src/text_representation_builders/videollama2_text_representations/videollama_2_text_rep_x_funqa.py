import json
import sys
import os
from os.path import dirname, abspath
import shutil
from tqdm import tqdm
root = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root)
from configs.configure import funqa_test_humor_video_dir 
from src.data_loaders.fun_qa_loader import FunQA_DataLoader
from src.model_loaders.video_llama2_loader import VideoLlama2Loader
from src.text_representation_builders.utils import save_temporary_frames_from_video

def generate_text_representation_from_video(videollama2, video_path=""):
  save_temporary_frames_from_video(video_path,output_path="./videollama_temp_frames")
  temp_video_frames_dir = "./videollama_temp_frames"
  video_frames = [os.path.join(temp_video_frames_dir,f) for f in os.listdir(temp_video_frames_dir) if f.endswith('.png')]
  sorted_video_frames = sorted(video_frames,key = lambda x: int(x.split("/")[-1].split(".")[0]))
  text_input = "What is happeing in the image? Instruction: answer within one line and cover all the details"
  text_rep = ""
  for i,frame in enumerate(sorted_video_frames):
    text_rep += f"{i}.0s: "
    text_rep += videollama2.infer(video_path=None, gr_img = frame,text_input=text_input)+"\n"
    print(text_rep)
    # delete temp frames inside the folder
  shutil.rmtree(temp_video_frames_dir)
  print(f"text rep : {text_rep}")
  return text_rep

def build_videollama2_text_rep(data = '',model = "",output_path =""):
  videollama2_text_rep_x_funqa = {}
  if os.path.exists(output_path):
    with open(output_path,'r') as f:
      videollama2_text_rep_x_funqa = json.load(f)
      print(f"videollama2_text_rep_x_funqa loaded with {len(videollama2_text_rep_x_funqa)} samples")
  for sample in tqdm(data):
    video_id = sample['visual_input']
    if video_id not in videollama2_text_rep_x_funqa:
      video_path = video_path = os.path.join(root,funqa_test_humor_video_dir, video_id)
      print(f"video_path: {video_path}")
      text_rep = generate_text_representation_from_video(model,video_path)
      sample["text_rep"] = text_rep
      videollama2_text_rep_x_funqa[video_id] = sample
      with open(output_path, 'w') as f:
        json.dump(videollama2_text_rep_x_funqa, f,indent=4)
        print(f"succesfully saved videollama2_text_rep_x_funqa with {len(videollama2_text_rep_x_funqa)} samples")





if __name__ == "__main__":
  funqa_humor_data = FunQA_DataLoader()
  videollama2 = VideoLlama2Loader()

  print(f"funqa humor test-set loaded with {len(funqa_humor_data)} videos")
  build_videollama2_text_rep(funqa_humor_data,videollama2,output_path = videollama2.args.output)
