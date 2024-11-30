import os
import sys
from os.path import dirname, abspath
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# Add root to sys.path
root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
from configs.configure import llama3_model_id


class Llama3Loader():
    def __init__(self) -> None:
        
        self.tokenizer, self.model = self.setup_llama_pipeline()

    # @memory.cache
    def setup_llama_pipeline(self):
      '''
      setup the llama pipeline
      '''
      # model_id = "./meta-llama/Meta-Llama-3-8B-Instruct"
      model_id =os.path.join(root, llama3_model_id)

      tokenizer = AutoTokenizer.from_pretrained(model_id)
      model = AutoModelForCausalLM.from_pretrained(model_id,
          torch_dtype=torch.bfloat16,
          device_map="auto",
      )
      return tokenizer, model
    
    def infer(self,content):
      messages = [
    {"role": "system", "content": "You are a video analyst. You can read the the video text representation and infer the start and end time of a given activity from the cue words found in the video text representation."},
    {"role": "user", "content": f"{content}"},
]
      input_ids = self.tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(self.model.device)
      terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
      outputs = self.model.generate(
      input_ids,
      max_new_tokens=500,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )
      response = outputs[0][input_ids.shape[-1]:]
      # print(self.tokenizer.decode(response, skip_special_tokens=True))
      generated_text = self.tokenizer.decode(response, skip_special_tokens=True)

      return generated_text
    

if __name__ == "__main__":
  llama3 = Llama3Loader()
  content = """The following are description of activities from frames extracted 1 second apart from a video clip: 
    0.0s: In the image, a young boy is being tickled by his father on a bed..\n1.0s: In the image, a young boy is playing with his father on a bed, jumping and running around while the father tries to catch him..\n2.0s: In the image, a man is performing a burpee exercise on a bed, jumping from a lying position to a standing position while simultaneously raising his arms above his head and kicking his feet forward..\n3.0s: In the image, a young girl is sitting on a bed, playing with a stuffed animal while a woman is standing behind her, holding a quilt and smiling ..\n

    The action [A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall.] has occurred in the video clip. What interval is the action most likely to start and end?
    Provide your best guess by providing the start and end time in json format
    """
  answer = llama3.infer(content)
  print(answer)
