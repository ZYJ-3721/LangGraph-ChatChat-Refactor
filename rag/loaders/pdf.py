import pymupdf
import importlib
from tqdm import tqdm
from typing import Literal
from unstructured.partition.text import partition_text
from langchain_community.document_loaders import UnstructuredFileLoader

from rag.loaders.img import IMAGE_LOADER_METHODS


class PDFLoader(UnstructuredFileLoader):
    def __init__(self, file_path, method: Literal["rapidocr", "llmcaption", "blipcaption"]="rapidocr", **kwargs):
        super().__init__(file_path, **kwargs)
        module = importlib.import_module("rag.loaders.img")
        ImageLoader = getattr(module, IMAGE_LOADER_METHODS[method])
        self.image_loader = ImageLoader()
    def pdf2text(self, file_path):
        content = ""
        pdf = pymupdf.open(file_path)
        with tqdm(total=len(pdf), position=0, leave=True, desc="PDF Loading") as pbar:
            for i, page in enumerate(pdf):
                text = page.get_text() # 获取每页文本
                content += text.replace("\n", "") + "\n"
                images = page.get_images() # 获取每页图片
                with tqdm(total=len(images), position=0, leave=False,
                        desc=f"PDF Loading With Image Processing (Page {i+1}/{len(pdf)})") as pbar2:
                    for image_info in images:
                        xref, index, width, height = image_info[:4]
                        if width > 10 and height > 10:
                            image_bytes = pdf.extract_image(xref)["image"]
                            if image_content := self.image_loader.img2text(image_bytes):
                                image_content = image_content.replace("\n", " ")
                                content += f"这里有一张图片内容为：“{image_content}”。\n"
                        pbar2.update()
                pbar.update()
        return content
    def _get_elements(self):
        text = self.pdf2text(self.file_path)
        return partition_text(text=text, **self.unstructured_kwargs)
