from os.path import dirname, abspath
import json
import os 
import sys

# Add root to sys.path
root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

# imports 
from configs.configure import charades_sta_test_path #"data/charades_sta/Charades_STA_test.json"

class Charades_STA_Dataloader():
  def __init__(self):
    self.data = self.load_data()
  def load_data(self):
    with open(os.path.join(root, charades_sta_test_path), 'r') as f:
      data = json.load(f)
    return data

  def __iter__(self):
    for video_info in self.data:
      yield video_info
  def __len__(self):
    return len(self.data)

if __name__ == "__main__":
  charades_sta = Charades_STA_Dataloader()
  print(f"Total videos in Charades STA: {len(charades_sta)}")
  for video_info in charades_sta:
    print(video_info)
    print(video_info['video_id'])
    print(type(video_info))
    break

