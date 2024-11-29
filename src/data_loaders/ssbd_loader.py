import json
import os
import sys

from os.path import dirname, abspath
root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

from configs.configure import ssbd_data_path,ssbd_video_dir

class Ssbd_DataLoader():
    def __init__(self):
        self.data = self.load_data()
        
    def load_data(self):
        with open(os.path.join(root,ssbd_data_path), 'r') as f:
            data = json.load(f)
        return data

    def __iter__(self):
        video_dir = os.path.join(root,ssbd_video_dir)
        for sample in self.data:
            video_tag = sample[0]
            behaviour_id = sample[1]['id']
            video_id = video_tag+'_'+str(behaviour_id)
            video_info = {}
            video_info = {'video_path': os.path.join(video_dir,video_tag+'.mp4'), 'behaviour': sample[1]}
            yield video_id,video_info
       
    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
   ssbd_data = Ssbd_DataLoader()
   print(f"Total videos in SSBD: {len(ssbd_data)}")
   for data in ssbd_data:
       print(data)
       break
   
