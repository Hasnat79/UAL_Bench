import os
from os.path import dirname, abspath
import sys
import json
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
root = dirname(dirname(dirname(abspath(__file__))))
video_chat2_root = os.path.join(root, "foundation_models", "Ask-Anything", "video_chat2")
sys.path.append(root)
sys.path.append(os.path.join(root, "foundation_models", "Ask-Anything", "video_chat2"))
from conversation import Chat
from tqdm import tqdm
# videochat
import torch
from utils.config import Config
from utils.easydict import EasyDict
from models import VideoChat2_it_vicuna  as VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType

# updating checkpoint paths in the config file and saving it
config_file = os.path.join(root,"configs/video_chat2_config.json")
with open(config_file, "r") as f:
    cfg = EasyDict(json.load(f))
cfg['model']["vit_blip_model_path"] = os.path.join(video_chat2_root,"umt_l16_qformer.pth")
cfg['model']['llama_model_path'] = os.path.join(video_chat2_root,"vicuna-7b-v0")
cfg['model']['videochat2_model_path'] = os.path.join(video_chat2_root,"videochat2_7b_stage2.pth")
with open(config_file, "w") as f:
    json.dump(cfg, f, indent=4)

class VideoChat2Loader():
    def __init__(self):
        self.chat = self.init_model()
    def __call__(self):
        return self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len
    @memory.cache
    def init_model():
        print('Initializing VideoChat')
        # config_file = os.path.join(root,"configs/video_chat2_config.json")
        cfg = Config.from_file(config_file)
        cfg.model.vision_encoder.num_frames = 4
        # cfg.model.videochat2_model_path = ""
        # cfg.model.debug = True
        model = VideoChat2_it(config=cfg.model)
        model = model.to(torch.device(cfg.device))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        videochat2_7b_stage3_path = os.path.join(video_chat2_root,"videochat2_7b_stage3.pth")
        state_dict = torch.load(videochat2_7b_stage3_path, "cpu")
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model = model.eval()

        chat = Chat(model)
        print('Initialization Finished')
        return chat
    def infer(self,video_path, user_message):
        num_beams = 1
        temperature = 0.1
        num_segments = 4 

        chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
        img_list = []
        llm_message, img_list, chat_state = self.chat.upload_video(video_path, chat_state, img_list, num_segments)
        chat_state =  self.chat.ask(user_message, chat_state)
        llm_message,llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "")

        # reset 
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        print(f"video_path: {video_path}")
        print(f"prompt: {user_message}")
        print(f"Answer: {llm_message}")
        return llm_message
    
if __name__ == "__main__":
    video_chat2 = VideoChat2Loader()
    print("Model loaded")
    video_path = root+"/data/uag_oops/videos/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"
    user_message = "What is happening in this video?"
    answer = video_chat2.infer(video_path, user_message)
    print(f"Answer: {answer}")
