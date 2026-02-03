#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import io
import sys
import threading
import pdfplumber

from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizerDocLayoutYOLO as LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer
from .engine import VietDocEngine

# New Phase-Based Architecture
from .pipeline import DocumentPipeline
from .phases import (
    LayoutAnalysisPhase,
    TextDetectionPhase,
    TextRecognitionPhase,
    PostProcessingPhase,
    DocumentReconstructionPhase,
)
from .implementations import (
    DocLayoutYOLOAnalyzer,
    PaddleOCRTextDetector,
    VietOCRRecognizer,
    SVTRv2Recognizer,
    LandingAIRecognizer,
    VietnameseTextPostProcessor,
    SmartMarkdownReconstruction,
    create_default_pipeline,
    create_svtrv2_pipeline,
    create_experimental_pipeline,
)


LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def init_in_out(args):
    from PIL import Image
    import os
    import traceback
    from .utils.file_utils import traversal_files
    images = []
    outputs = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    def pdf_pages(fnm, rel_path, zoomin=2):
        nonlocal outputs, images
        with sys.modules[LOCK_KEY_pdfplumber]:
            pdf = pdfplumber.open(fnm)
            # FIX: Use extend instead of assignment to accumulate pages from all PDFs
            new_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                                enumerate(pdf.pages)]
            images.extend(new_images)

        for i in range(len(new_images)):
            outputs.append(rel_path + f"_{i}.jpg")
        pdf.close()

    def images_and_outputs(fnm, rel_path):
        nonlocal outputs, images
        if fnm.split(".")[-1].lower() == "pdf":
            pdf_pages(fnm, rel_path)
            return
        try:
            fp = open(fnm, 'rb')
            binary = fp.read()
            fp.close()
            images.append(Image.open(io.BytesIO(binary)).convert('RGB'))
            outputs.append(rel_path)
        except Exception:
            traceback.print_exc()

    if os.path.isdir(args.inputs):
        for fnm in traversal_files(args.inputs):
            rel_path = os.path.relpath(fnm, args.inputs)
            images_and_outputs(fnm, rel_path)
    else:
        images_and_outputs(args.inputs, os.path.basename(args.inputs))

    for i in range(len(outputs)):
        outputs[i] = os.path.join(args.output_dir, outputs[i])

    return images, outputs


__all__ = [
    # Legacy API (backward compatibility)
    "OCR",
    "Recognizer",
    "LayoutRecognizer",
    "TableStructureRecognizer",
    "VietDocEngine",
    "init_in_out",

    # New Phase-Based Architecture
    "DocumentPipeline",

    # Abstract Phase Interfaces
    "LayoutAnalysisPhase",
    "TextDetectionPhase",
    "TextRecognitionPhase",
    "PostProcessingPhase",
    "DocumentReconstructionPhase",

    # Concrete Implementations
    "DocLayoutYOLOAnalyzer",
    "PaddleOCRTextDetector",
    "VietOCRRecognizer",
    "SVTRv2Recognizer",
    "LandingAIRecognizer",
    "VietnameseTextPostProcessor",
    "SmartMarkdownReconstruction",

    # Factory Functions
    "create_default_pipeline",
    "create_svtrv2_pipeline",
    "create_experimental_pipeline",
]
