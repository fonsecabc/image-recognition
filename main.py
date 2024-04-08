import os
from dotenv import load_dotenv, find_dotenv
from langchain.indexes import VectorstoreIndexCreator
from modules.video_extractor import extract_frames_from_video
from langchain_community.document_loaders import ImageCaptionLoader

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
execution_path = os.getcwd()

video_path = os.path.join(execution_path, 'videos/video.mov')
output_dir = os.path.join(execution_path, 'videos')
frame_rate = 64

video_data_path = extract_frames_from_video(video_path, output_dir, frame_rate)
if isinstance(video_data_path, Exception):
  print(video_data_path)
  exit()

#image_path = os.path.join(execution_path, 'images/image.png')
#loader = ImageCaptionLoader(images=[image_path])

images = os.listdir(video_data_path)
images_paths = [os.path.join(video_data_path, image) for image in images]
print(images_paths)
loader = ImageCaptionLoader(images_paths)

list_docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
result = index.query('These images are frames from a video, describre the conjunction of these frames as a single video.')
print(result)