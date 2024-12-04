'''
we will run llama3 to localize (predict start and end time) unusual activity from the blip2 text representation from the uag oops dataset  
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

def extract_time_from_answer(answer):
    '''
    extract start and end time from the answer
    expected format : {  "start_time": 7.0,
                        "end_time": 7.0
                      }
    '''
    pattern_1 = r'"start_time": (\d+\.\d+)(s)?,\s*"end_time": (\d+\.\d+)(s)?'
    times = re.findall(pattern_1, answer)
    print(times)
    
    if times: 
        pred_start = float(times[0][0])
        pred_end = float(times[0][2])
    else: 
        pred_start = None
        pred_end = None
    return pred_start, pred_end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text representations for videos using BLIP2 model.")
    parser.add_argument('--input', type=str, required=True, help='Input JSON file name: blip2 text representation file from funqa dataset')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file name')
    args = parser.parse_args()
    
    llama3 = Llama3Loader()
    # load the blip2 text representation from funqa dataset
    with open(args.input) as f:
        blip2_text_rep_x_funqa = json.load(f)
    
    print(f"total number of videos in funqa text_rep: {len(blip2_text_rep_x_funqa)}")
    

    results = {}
    if os.path.exists(args.output):
      with open(args.output, 'r') as f:
          results = json.load(f)
      print(f"Loaded {len(results)} results")
    
        


    # run llama3 for all videos
    for video_id, video_info in tqdm(blip2_text_rep_x_funqa.items()):
        if video_id not in results:
            print(f"video_id: {video_id}")
            video_text_rep = video_info["text_rep"]
            # print(f"video_text_rep: {video_text_rep}")
            query = video_info["instruction"]
            content = f"""Find the start time and end time of the query below given the video text representation. Even if the query is not present in the description, try to find relationship between the meaning of words and infer. You must predict an answer and do not predict any null prediction. Give your answer in json format.
Query: {query}
Video Text Representation: {video_text_rep}
"""    
            try : 
              llama3_generate_text = llama3.infer(content)
              video_info["llama3_pred"] = llama3_generate_text
              pred_start, pred_end = extract_time_from_answer(llama3_generate_text)
              video_info["pred_start"] = pred_start
              video_info["pred_end"] = pred_end
              results[video_id] = video_info
              with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in {args.output} with {len(results)} samples")
            except Exception as e:
              print(e)
            
    # check the results
    print("verifying results")
    with open(args.output, 'r') as f:
        results = json.load(f)
        print(f"size of results: {len(results)}")
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            none_count +=1
    print(f"none_count: {none_count}")#83








    