from os.path import dirname, abspath
import json
import os
import sys
# Add root to sys.path
root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

from configs.configure import uag_oops_dataset_path, uag_oops_video_dir


class UagOopsV1_DataLoader():
    def __init__(self):
        self.data = self.load_data()
    def load_data(self):
        with open(os.path.join(root,uag_oops_dataset_path), 'r') as f:
            data = json.load(f)
        return data

    def __iter__(self):
        video_dir = os.path.join(root,uag_oops_video_dir)
        for video_id,video_info in self.data.items():
            video_info['video_path'] = os.path.join(video_dir,video_id+'.mp4')
            yield video_id,video_info
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    uag_oops_v1 = UagOopsV1_DataLoader()
    print(f"Total videos in UAG OOPS V1: {len(uag_oops_v1)}")
    for video_id, video_info in uag_oops_v1:
        print(video_id, video_info)
        break
        