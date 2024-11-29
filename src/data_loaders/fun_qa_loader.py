from os.path import dirname, abspath
import json
import os
import sys
# Add root to sys.path
root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

from configs.configure import funqa_data_path


class FunQA_DataLoader():
    def __init__(self) -> None:
        self.data = self.load_data()
        # self.save_data(self.data,funqa_data_path)
    def save_data(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f,indent =4)
    def filter_humor_annotations(self, data) -> list:
        humor_data = []
        for video_info in data:
            if video_info['task'] == "H1":
                humor_data.append(video_info)
        print(f" length of humor_data: {len(humor_data)}")
        return humor_data
    def filter_unique_video_data(self, data) -> list:
        unique_data = []
        video_ids = set()
        for video_info in data:
            video_id = video_info['visual_input']
            if video_id not in video_ids:
                video_ids.add(video_id)
                unique_data.append(video_info)
        print(f" length of unique_data: {len(unique_data)}")
        return unique_data
    def extract_ground_truth_start_end_time(self, data) -> list:
        # current ground truth are in key 'output' and they are the start frame and end frame, fps is 30
        for video_info in data:
            # print("video_info: ",video_info)
            # print(video_info['output'].split(","))
            output_list = [int(video_info['output'].split(",")[0][1:]),int(video_info['output'].split(",")[1][1:-1])]
            video_info['start_time'] = round(output_list[0]/30,2)
            video_info['end_time'] = round(output_list[1]/30,2)
        return data
    def load_data(self):
        with open(os.path.join(root, funqa_data_path), 'r') as f:
            data = json.load(f) #list
        # print(f" length of data: {len(data)}")
        
        return data
    
    ''' this function is used to load the original funqa annotation file and proces to the uag_funqa dataset'''
    # def __load_data(self):
    #     with open(funqa_test_annotations_h1_path, 'r') as f:
    #         data = json.load(f) #list
    #     print(f" length of data: {len(data)}")
        
    #     data = self.filter_unique_video_data(data)
    #     #  extract video informations that has task : "H1"
    #     data = self.filter_humor_annotations(data)

    #     # extract ground truth start and end time
    #     data = self.extract_ground_truth_start_end_time(data)
    #     return data
    def __iter__(self):
        for video_info in self.data:
            yield video_info
    
    def __len__(self):
        return len(self.data)




if __name__ == "__main__":
    funqa_test_humor = FunQA_DataLoader()
    for video_info in funqa_test_humor:
        print(video_info)
        break
#  length of data: 30170
#  length of unique_data: 424
#  length of humor_data: 172
# {'instruction': "Identify the video's funny moment.", 'visual_input': 'H_A_101_1433_1631.mp4', 'output': '[0000,  0195]', 'task': 'H1', 'ID': 'test_0', 'start_time': 0, 'end_time': 6}
