import json
import argparse
import numpy as np



def calculate_iou(gt_start,gt_end,pred_start,pred_end):
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    iou = 1.0* (intersection/union)
    return iou
def calculate_abs_dist(gt_start,gt_end , pred_start, pred_end):
    return abs(gt_start - pred_start)+ abs(gt_end - pred_end)
def eval_recall_1_iou(result: list): 
  sample_len = len(result)
  ious =[]
  toggle=True
  IoU_threshold = [0.3 ,0.5, 0.7]
  for threshold in IoU_threshold:
      correct_count = 0
      for video in result:
          gt_start,gt_end = video['start_time'],video['end_time']

          duration = gt_end
          if video['pred_start'] is None or video['pred_end'] is None:
              #if either pred_start or pred_end is None, iou = 0
              iou = 0.0
          else:
              pred_start,pred_end = video['pred_start'],video['pred_end']
              iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
          if toggle: 
              ious.append(iou)
          if iou >= threshold:
              correct_count += 1
      toggle = False
      iou_recall_top_1 = 100*(correct_count / sample_len)
      print(f'correct_count: {correct_count} len(result): {sample_len}')
      print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}; mIoU: {(np.mean(ious)):.2f}")

def abs_dist(result: list):
  sample_len = len(result)
  abs_distances = []
  toggle = True
  threshold_seconds = [0,1,3,5,7]
  for threshold in threshold_seconds:
      correct_count = 0
      for video in result:
          gt_start,gt_end = video['start_time'],video['end_time']
          duration = gt_end
          if video['pred_start'] is None or video['pred_end'] is None:
              # if either pred_start or pred_end is None, abs_dist = 0 
              abs_distance_between_gt_pred = duration
          else:
              pred_start,pred_end = video['pred_start'],video['pred_end']
              abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
          if abs_distance_between_gt_pred <= threshold:
              correct_count += 1
          if toggle:
              if abs_distance_between_gt_pred > 10000:
                  print(f"video{video['video_id']} abs_distance_between_gt_pred: {abs_distance_between_gt_pred}")
                  abs_distances.append(duration)
              else:
                  abs_distances.append(abs_distance_between_gt_pred)


      toggle = False
      abs_recall_top_1 = 100*(correct_count / sample_len)
      print(f"correct_count: {correct_count} len(result): {sample_len}")
      print(f"Threshold m = {threshold}s R@1: {abs_recall_top_1:.2f} mean abs distances: {np.mean(abs_distances):.2f}")

def accuracy_pred_start(result: list):
    sample_len = len(result)
    correct_within_1_sec_count = 0
    correct_within_quarter_sec_count = 0
    for video in result:
        gt_start,gt_end = video['start_time'],video['end_time']
        duration = gt_end
            
        if video['pred_start'] is None and video['pred_end'] is None:
            # if both pred_start and pred_end are None, generate random prediction for both
            correct_within_1_sec_count+=0
            correct_within_quarter_sec_count+=0
            continue
        elif video['pred_start'] is None and video['pred_end'] is not None:
            # if pred_start is None, generate random prediction for pred_start
            correct_within_1_sec_count+=0
            correct_within_quarter_sec_count+=0
            continue
        elif video['pred_end'] is None and video['pred_start'] is not None:
            # if pred_end is None, generate random prediction for pred_end
            correct_within_1_sec_count+=0
            correct_within_quarter_sec_count+=0
            continue
        else:
            pred_start,pred_end = video['pred_start'],video['pred_end']
        # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
        
        if abs(gt_start - pred_start) <= 1:
            correct_within_1_sec_count+=1
        if abs(gt_start - pred_start) <= 0.25:
            correct_within_quarter_sec_count+=1

    print(f"correct_within_1_sec_count: {correct_within_1_sec_count} len(result): {sample_len}")
    print(f"correct_within_quarter_sec_count: {correct_within_quarter_sec_count} len(result): {sample_len}")
    # accuracy
    accuracy_within_1_sec = 100*(correct_within_1_sec_count / sample_len)
    accuracy_within_quarter_sec = 100*(correct_within_quarter_sec_count / sample_len)
    print(f"Accuracy within 1 sec: {accuracy_within_1_sec:.2f}")
    print(f"Accuracy within 0.25 sec: {accuracy_within_quarter_sec:.2f}")

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Process a JSON file')
  parser.add_argument('--results_file', type=str, help='Path to the input JSON file')
  args = parser.parse_args()

  with open(args.results_file, 'r') as f:
    results = json.load(f)
  print("==========recal@1 iou >= m======================")
  eval_recall_1_iou(results)
  print("==========abs dist <= m======================")
  abs_dist(results)
  print("============Onset (start_time) prediction accuracy====================")
  accuracy_pred_start(results)




  

  
