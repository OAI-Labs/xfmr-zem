import os
import abc
from typing import Dict, Any, List
from PIL import Image
from loguru import logger

class OCREngineBase(abc.ABC):
    """
    Abstract Base Class for OCR Engines (Dependency Inversion & Open/Closed).
    """
    @abc.abstractmethod
    def process(self, image_path: str) -> Dict[str, Any]:
        """Process an image and return extracted text and metadata."""
        pass

class TesseractEngine(OCREngineBase):
    """
    Lightweight OCR using Tesseract (Fast & Simple).
    """
    def __init__(self):
        try:
            import pytesseract
            import shutil
            
            # Check if tesseract binary exists
            if not shutil.which("tesseract"):
                raise RuntimeError(
                    "Tesseract binary not found. To use the 'tesseract' engine, "
                    "please install it using: sudo apt install tesseract-ocr"
                )
            
            self.pytesseract = pytesseract
        except ImportError:
            logger.error("pytesseract not installed. Please install with 'pip install pytesseract'")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using Tesseract to process: {image_path}")
        image = Image.open(image_path)
        text = self.pytesseract.image_to_string(image)
        return {
            "text": text,
            "engine": "tesseract",
            "metadata": {"format": image.format, "size": image.size}
        }

class PaddleEngine(OCREngineBase):
    """
    Medium-weight OCR using PaddleOCR (High accuracy for multi-language).
    """
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en') # Default to English
        except ImportError:
            logger.error("paddleocr not installed. Please install with 'pip install paddleocr paddlepaddle'")
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using PaddleOCR to process: {image_path}")
        result = self.ocr.ocr(image_path, cls=True)
        
        full_text = []
        scores = []
        for line in result:
            if line:
                for res in line:
                    full_text.append(res[1][0])
                    scores.append(float(res[1][1]))
        
        return {
            "text": "\n".join(full_text),
            "engine": "paddleocr",
            "metadata": {"avg_confidence": sum(scores)/len(scores) if scores else 0}
        }

class HuggingFaceVLEngine(OCREngineBase):
    """
    Advanced OCR using Hugging Face Vision Language Models (e.g. Qwen2-VL, Molmo).
    """
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id or "Qwen/Qwen2-VL-2B-Instruct"
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            try:
                import torch
                from transformers import AutoModelForVision2Seq, AutoProcessor
                
                logger.info(f"Loading Hugging Face VL model: {self.model_id} (this may take a while)...")
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                # Using AutoModelForVision2Seq for generality
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
            except ImportError:
                logger.error("transformers/torch not installed. Required for HuggingFace-VL.")
                raise
            except Exception as e:
                logger.error(f"Error loading model {self.model_id}: {e}")
                raise

    def process(self, image_path: str) -> Dict[str, Any]:
        self._lazy_load()
        logger.info(f"Using {self.model_id} via HuggingFaceVLEngine to process: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Generic prompt for OCR
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|vision_end|>Extract all text from this image exactly as it appears.<|im_end|>\n<|im_start|>assistant\n"
        
        # Note: Inference logic might vary slightly between models, but we use a common VLM pattern
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up text if it contains the prompt
        if "assistant\n" in text:
            text = text.split("assistant\n")[-1]
            
        return {
            "text": text,
            "engine": "huggingface",
            "metadata": {"model_id": self.model_id}
        }

class VietOCREngine(OCREngineBase):
    """
    Specialized Vietnamese OCR using built-in Deep-ocr DocumentPipeline (Layout + OCR + MD).
    """
    def __init__(self):
        try:
            from xfmr_zem.servers.ocr.deepdoc_vietocr.pipeline import DocumentPipeline
            from xfmr_zem.servers.ocr.deepdoc_vietocr.implementations import (
                PaddleStructureV3Analyzer,
                PaddleOCRTextDetector,
                VietOCRRecognizer,
                VietnameseTextPostProcessor,
                SmartMarkdownReconstruction
            )
            
            logger.info("Initializing Internal Deep-ocr DocumentPipeline for Vietnamese...")
            self.pipeline = DocumentPipeline(
                layout_analyzer=PaddleStructureV3Analyzer(),
                text_detector=PaddleOCRTextDetector(),
                text_recognizer=VietOCRRecognizer(),
                post_processor=VietnameseTextPostProcessor(),
                reconstructor=SmartMarkdownReconstruction()
            )
        except Exception as e:
            logger.error(f"Error loading internal Deep-ocr components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def process(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Using Internal Deep-ocr (DocumentPipeline) to process: {image_path}")
        from PIL import Image
        
        img = Image.open(image_path)
        
        # document.process returns reconstructed markdown text
        markdown_text = self.pipeline.process(img)
            
        return {
            "text": markdown_text,
            "engine": "deepdoc_vietocr",
            "metadata": {"format": "markdown"}
        }

class OCREngineFactory:
    """
    Factory to create OCR engines (Switching strategy).
    """
    @staticmethod
    def get_engine(engine_type: str, **kwargs) -> OCREngineBase:
        if engine_type == "tesseract":
            return TesseractEngine()
        elif engine_type == "paddle":
            return PaddleEngine()
        elif engine_type == "huggingface" or engine_type == "qwen":
            return HuggingFaceVLEngine(model_id=kwargs.get("model_id"))
        elif engine_type == "viet":
            return VietOCREngine()
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
