import os
import yaml
from os.path import dirname, abspath
import sys
import argparse
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
root = dirname(dirname(dirname(abspath(__file__))))
video_llama_root = os.path.join(root, "foundation_models", "Video-LLaMA")
sys.path.append(root)
sys.path.append(video_llama_root)
import torch.backends.cudnn as cudnn
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

with open(os.path.join(video_llama_root, "eval_configs/video_llama_eval_withaudio.yaml"), "r") as f:
    video_llama_eval_withaudio = yaml.safe_load(f)

# updating checkpoints in the eval config of videollama

video_llama_eval_withaudio['model']['llama_model'] = os.path.join(video_llama_root,"Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf")
video_llama_eval_withaudio['model']['imagebind_ckpt_path'] = os.path.join(video_llama_root,"Video-LLaMA-2-7B-Finetuned")
video_llama_eval_withaudio['model']['ckpt'] = os.path.join(video_llama_root,'Video-LLaMA-2-7B-Finetuned/VL_LLaMA_2_7B_Finetuned.pth')
video_llama_eval_withaudio['model']['ckpt_2'] = os.path.join(video_llama_root,'Video-LLaMA-2-7B-Finetuned/AL_LLaMA_2_7B_Finetuned.pth')
with open(os.path.join(video_llama_root, "eval_configs/video_llama_eval_withaudio.yaml"), "w") as f:
    yaml.dump(video_llama_eval_withaudio, f)


class VideoLlama2Loader():
    def __init__(self):
        self.args = self.parse_args()
        self.chat = self.initialize_model(self.args)

    def infer(self,video_path,gr_img, text_input, audio_flag=True, num_beams=1, temperature=0.5):
      chat_state = None
      img_list = None
      
      
      chatbot = []
      
      chat_state, img_list, chatbot = self.upload_imgorvideo(video_path, gr_img, text_input, chat_state, chatbot, audio_flag)
      chat_state, chatbot, chat_state = self.gradio_ask(text_input, chatbot, chat_state)
      chatbot, chat_state, img_list = self.gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
      
      return chatbot[-1][1]

    
    def upload_imgorvideo(self,gr_video, gr_img, text_input, chat_state, chatbot, audio_flag):
      if self.args.model_type == 'vicuna':
          chat_state = default_conversation.copy()
      else:
          chat_state = conv_llava_llama_2.copy()
      if gr_img is None and gr_video is None:
          return None, None, None, chat_state, None
      elif gr_img is not None and gr_video is None:
          print(gr_img)
          chatbot = chatbot + [((gr_img,), None)]
          chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
          img_list = []
          llm_message = self.chat.upload_img(gr_img, chat_state, img_list)
          return chat_state, img_list, chatbot
      elif gr_video is not None and gr_img is None:
          # print(gr_video)
          chatbot = chatbot + [((gr_video,), None)]
          chat_state.system =  ""
          img_list = []
          if audio_flag:
              llm_message = self.chat.upload_video(gr_video, chat_state, img_list)
          else:
              llm_message = self.chat.upload_video_without_audio(gr_video, chat_state, img_list)
          return chat_state, img_list, chatbot
      else:
          return chat_state, None, chatbot
      

    def gradio_ask(self,user_message, chatbot, chat_state):
      if len(user_message) == 0:
          return chat_state, chatbot, chat_state
      self.chat.ask(user_message, chat_state)
      chatbot = chatbot + [[user_message, None]]
      return chat_state, chatbot, chat_state

    def gradio_answer(self,chatbot, chat_state, img_list, num_beams, temperature):
        llm_message = self.chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=500)[0]
        chatbot[-1][1] = llm_message
        # print(chat_state.get_prompt())
        # print(chat_state)
        return chatbot, chat_state, img_list  
    @memory.cache
    def initialize_model(args):
        print('Initializing Chat')
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print('Initialization Finished')
        return chat
    def parse_args(self):
      parser = argparse.ArgumentParser(description="Demo")
      cfg_path =os.path.join( root , "foundation_models/Video-LLaMA/eval_configs/video_llama_eval_withaudio.yaml")
      parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
      parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
      parser.add_argument("--model_type", type=str, default='llama2', help="The type of LLM")
      parser.add_argument(
          "--options",
          nargs="+",
          help="override some settings in the used config, the key-value pair "
          "in xxx=yyy format will be merged into config file (deprecate), "
          "change to --cfg-options instead.",
      )
      parser.add_argument('--output', type=str, required=True, help='Output JSON file name')
      args = parser.parse_args()
      return args

if __name__ == "__main__":
    video_llama2 = VideoLlama2Loader()
    print("Model loaded")
    video_path = root+"/data/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"
    query = "What is happening in this video?"
    answer = video_llama2.infer(video_path,gr_img=None,text_input= query)
    print(f"Answer: {answer}")
    # Model loaded
    # Answer: The person is arm flapping.
