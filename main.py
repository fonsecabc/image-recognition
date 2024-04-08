from langchain_community.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

execution_path = os.getcwd()
image_path = os.path.join(execution_path, 'images/image.png')
loader = ImageCaptionLoader(images=[image_path])
list_docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
result = index.query('Describe this image in 2 sentences.') 
print(result)
