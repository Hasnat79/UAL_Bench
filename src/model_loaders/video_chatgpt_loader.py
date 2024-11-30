import os
from os.path import dirname, abspath
import sys
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)

root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)
sys.path.append(os.path.join(root, "foundation_models", "Video-ChatGPT"))
from configs.configure import llava_lightning_7b_v1_1_path, video_chatgpt_weights_path
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer



class VideoChatGPTLoader():

    def __init__(self):
        # args = self.parse_args()
        llava_path = os.path.join(root, llava_lightning_7b_v1_1_path)
        weights_path = os.path.join(root, video_chatgpt_weights_path)
        self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(llava_path, weights_path)
        
    def __call__(self):
        return self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len
    def get_video_frames (self, video_path, num_frm=100):
        print(f"Loading video from {video_path}")
        return load_video(video_path,num_frm=num_frm)
      
    def infer(self,video_frames,query):
        conv_mode = "video-chatgpt_v1"
        answer = video_chatgpt_infer(video_frames, query, conv_mode, self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len)
        print(f"Answer: {answer}")
        return answer
    

if __name__ == "__main__":
    video_chatgpt = VideoChatGPTLoader()
    print("Model loaded")
    video_path = root+"/data/uag_oops/videos/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"
    video_frames = video_chatgpt.get_video_frames(video_path)
    query = "What is happening in this video?"
    answer = video_chatgpt.infer(video_frames, query)
    print(f"Answer: {answer}")

