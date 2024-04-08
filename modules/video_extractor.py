import cv2
import os

def extract_frames_from_video(video_path: str, output_dir: str, frame_rate: int) -> str or Error:
  video = cv2.VideoCapture(video_path)
  video_name = os.path.basename(video_path)
  video_data_path = os.path.join(output_dir, f'{video_name}_data')
  
  try :
    if not video.isOpened():
      raise Exception('Error opening video file')
    
    if not os.path.exists(video_data_path):
      os.makedirs(video_data_path)
    else: 
      files = os.listdir(video_data_path)
      for file in files:
        os.remove(os.path.join(video_data_path, file))

    frame_count = 0

    while True:
      ret, frame = video.read()
      if not ret:
        break

      if frame_count % frame_rate == 0:
        frame_path = f'{video_data_path}/frame_{frame_count}.png'
        cv2.imwrite(frame_path, frame)

      frame_count += 1

    return video_data_path
  except Exception as e:
    return e
