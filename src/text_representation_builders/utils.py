import cv2
import os

def load_frame(frame_path):
    '''loads a frame from a given frame path
    '''
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def save_temporary_frames_from_video(video_path="", output_path = "./temp_frames"):
    '''from a video path it saves the first frame and the rest of the frames at 1fps in a temp folder
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    os.system(f"""ffmpeg -i "{video_path}" -vf fps=1 {output_path}/%d.png""")
