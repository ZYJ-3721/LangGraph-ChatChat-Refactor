import re
import io
import cv2
import base64
import openai
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from unstructured.partition.text import partition_text
from langchain_community.document_loaders import UnstructuredFileLoader

LLM_CAPTION_MODEL = "glm-4v-flash"
LLM_CAPTION_API_KEY = "e7429a21a08a41b089a55dda1facfdb5.pYU0jXzarUrKNhBr"
LLM_CAPTION_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"


class RapidOCRImageLoader(UnstructuredFileLoader):
    def __init__(self, file_path=None, **kwargs):
        super().__init__(file_path, **kwargs)
        self.ocr = RapidOCR()
    def preprocess(self, image_bytes):
        """图片增强处理"""
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_array = cv2.medianBlur(img_array, ksize=3)
        _, img_array = cv2.imencode('.png', img_array)
        return img_array.tobytes()
    def postprocess(self, text):
        """文本清洗处理"""
        text = re.sub(r'[^\w.,。，]{1,}', ' ', text)
        text = re.sub(r'[A-Z\s]{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    def img2text(self, image_bytes):
        image_bytes = self.preprocess(image_bytes)
        results = self.ocr(image_bytes) # 识别文本
        if results[0] is None: return "" # 识别不到文本
        text = " ".join([line[1] for line in results[0]])
        return self.postprocess(text)
    def _get_elements(self):
        with open(self.file_path, "rb") as f:
            image_bytes = f.read()
        text = self.img2text(image_bytes)
        return partition_text(text=text, **self.unstructured_kwargs)


class LLMCaptionImageLoader(UnstructuredFileLoader):
    def __init__(self, file_path=None, base_url=LLM_CAPTION_BASE_URL,
                 api_key=LLM_CAPTION_API_KEY, model=LLM_CAPTION_MODEL, **kwargs):
        super().__init__(file_path, **kwargs)
        self.client = openai.Client(base_url=base_url, api_key=api_key)
        self.model = model
    def img2text(self, image_bytes):
        try:
            image_base64 = base64.b64encode(image_bytes).decode()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述图片"},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return "LLM描述图片失败！"
    def _get_elements(self):
        with open(self.file_path, "rb") as f:
            image_bytes = f.read()
        text = self.img2text(image_bytes)
        return partition_text(text=text, **self.unstructured_kwargs)


class BLIPCaptionImageLoader(UnstructuredFileLoader):
    def __init__(self, file_path=None, pretrained_model_name_or_path=BLIP_CAPTION_MODEL, **kwargs):
        super().__init__(file_path, **kwargs)
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.processor = BlipProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = BlipForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    def img2text(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)) 
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    def _get_elements(self):
        with open(self.file_path, "rb") as f:
            image_bytes = f.read()
        text = self.img2text(image_bytes)
        return partition_text(text=text, **self.unstructured_kwargs)


IMAGE_LOADER_METHODS = {
    "rapidocr": "RapidOCRImageLoader",
    "llmcaption": "LLMCaptionImageLoader",
    "blipcaption": "BLIPCaptionImageLoader",
}
