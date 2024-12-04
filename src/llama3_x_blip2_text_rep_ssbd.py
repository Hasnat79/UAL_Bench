'''
we will run llama3 to localize (predict start and end time) unusual activity from the blip2 text representation from ssbd dataset
'''

import os
from os.path import dirname, abspath
import sys
import argparse
import re
from tqdm import tqdm

root = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root)

import json
from model_loaders.llama3_loader import Llama3Loader


# def extract_time_from_answer(answer):
#     '''
#     extract start and end time from the answer
#     expected format : {  "start_time": 7.0,
#                         "end_time": 7.0
#                       }
#     '''
#     pattern_1 = r'"start_time": (\d+\.\d+)(s)?,\s*"end_time": (\d+\.\d+)(s)?'
#     times = re.findall(pattern_1, answer)
#     print(times)
    
#     if times: 
#         pred_start = float(times[0][0])
#         pred_end = float(times[0][2])
#     else: 
#         pred_start = None
#         pred_end = None
#     return pred_start, pred_end
def extract_time_from_answer(pred):
    pattern = r'"startTime": (\d+\.\d+),\s+"endTime": (\d+\.\d+)'
    pattern_2 = r'"start_time": (\d+\.?\d*),\s*"end_time": (\d+\.?\d*)'
    pattern_3 = r'Start time: (\d+\.?\d*)s\s+End time: (\d+\.?\d*)s'
    pattern_4 = r'Start Time: (\d+\.?\d*)s\s+End Time: (\d+\.?\d*)s'
    pattern_5 = r"start_time:\s*(\d+\.\d+),\s*end_time:\s*(\d+\.\d+)"

    match = re.search(pattern, pred)
    match_2 = re.search(pattern_2, pred)
    match_3 = re.search(pattern_3, pred)
    match_4 = re.search(pattern_4, pred)
    match_5 = re.search(pattern_5, pred)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    elif match_2:
        start_time = float(match_2.group(1))
        end_time = float(match_2.group(2))
        return start_time, end_time   
    elif match_3:
        start_time = float(match_3.group(1))
        end_time = float(match_3.group(2))
        return start_time, end_time
    elif match_4:
        start_time = float(match_4.group(1))
        end_time = float(match_4.group(2))
        return start_time, end_time
    elif match_5:
        start_time = float(match_5.group(1))
        end_time = float(match_5.group(2))
        return start_time, end_time
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text representations for videos using BLIP2 model.")
    parser.add_argument('--input', type=str, required=True, help='Input JSON file name: blip2 text representation file from funqa dataset')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file name')
    args = parser.parse_args()

    llama3 = Llama3Loader()
    # load the blip2 text representation from the ssbd
    with open(args.input) as f:
        blip2_text_rep_ssbd_dataset = json.load(f)
    
    print(f"total number of videos in the ssbd dataset v1: {len(blip2_text_rep_ssbd_dataset)}")

    results = {}
    if os.path.exists(args.output):
      with open(args.output, 'r') as f:
          results = json.load(f)
      print(f"Loaded {len(results)} results")

    # run llama3 for all videos
    for video_id, video_info in tqdm(blip2_text_rep_ssbd_dataset.items()):
        
        if video_id not in results or results[video_id]["text_rep"] == "":
            print(f"video_id: {video_id}")
            video_text_rep = video_info["text_rep"]
            category = video_info["behavior"]["category"]
            intensity = video_info["behavior"]["intensity"]
            content = f"""Find the start time and end time of the query below given the video text representation. Even if the query is not present in the description, try to find relationship between the meaning of words and infer. Give your answer in json format
Query: A person is {category} with {intensity} intensity.
Video Text Representation: {video_text_rep}
"""         
            try : 
              llama3_generate_text = llama3.infer(content)
              video_info["llama3_pred"] = llama3_generate_text
              pred_start, pred_end = extract_time_from_answer(llama3_generate_text)
              video_info["pred_start"] = pred_start
              video_info["pred_end"] = pred_end
              print(f"pred_start: {pred_start}, pred_end: {pred_end}")
              results[video_id] = video_info
              with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in {args.output} with {len(results)} samples")
            except Exception as e:
              print(e)
        break
            
            
            
    # check the results
    print("verifying results")
    with open(args.output, 'r') as f:
        results = json.load(f)
        print(f"size of results: {len(results)}")#99
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            none_count +=1
    print(f"none_count: {none_count}")#61








    