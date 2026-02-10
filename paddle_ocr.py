import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Disable PaddleX check for connectivity to model hosters (User Request)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# We import PaddleOCR inside the class or method to avoid hard dependency if library is missing during development/mocking
# But for production code, it should be at top level.
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRExecutionError(Exception):
    """Custom exception for errors during OCR execution."""
    pass

class PaddleOCRProcessor:
    """
    Adapter class for PaddleOCR (Python Library Version).
    Responsible for executing the OCR engine on a given image/PDF and returning the result.
    
    Implements Singleton pattern for the heavy model loading.
    """
    _engine_instance = None

    def __init__(self):
        """Initialize the processor (Lazy loads the model)."""
        self.output_dir = config.OUTPUT_DIR
        self._ensure_engine_loaded()

    @classmethod
    def _ensure_engine_loaded(cls):
        """Singleton loader for PaddleOCR engine to avoid re-init costs."""
        if cls._engine_instance is None:
            if PaddleOCR is None:
                raise ImportError("paddleocr library not installed. Please install specific paddleocr version.")
            
            logger.info(f"Initializing PaddleOCR Engine (GPU={config.USE_GPU}, AngleCls={config.USE_ANGLE_CLS}, Lang={config.LANG})...")
            
            try:
                # PaddleOCR 3.x initialization with modern parameters
                device_arg = "gpu" if config.USE_GPU else "cpu"
                
                logger.info(f"Initializing with device={device_arg}...")
                cls._engine_instance = PaddleOCR(
                    use_angle_cls=config.USE_ANGLE_CLS,
                    lang=config.LANG,
                    device=device_arg,
                    # Disable preprocessing models that cause cuDNN initialization errors
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False
                )
                logger.info("PaddleOCR Engine initialized successfully.")
            except Exception as e:
                raise OCRExecutionError(f"Failed to initialize PaddleOCR engine: {e}") from e

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Run PaddleOCR on the specified file using the loaded python library.

        Args:
            file_path (Path): Absolute path to the input image or PDF.

        Returns:
            Dict[str, Any]: The structured output.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        
        try:
            # Run OCR
            # PaddleOCR.ocr() handles PDF vs Image internally based on extension
            print(f"DEBUG OCR: Calling PaddleOCR.ocr() on {file_path}...")
            results = self._engine_instance.ocr(str(file_path))
            
            if not results:
                logger.warning(f"No text detected in {file_path}")
                return {
                    "source_file": str(file_path),
                    "raw_text_content": "",
                    "pages": [] 
                }

            # Parse Results based on file type
            pages_content = []
            
            # Heuristic: If PDF, results is usually [page1, page2, ...]
            # If Image, results is [line1, line2, ...]
            # We can check file extension or list depth
            is_pdf = file_path.suffix.lower() == '.pdf'
            
            if is_pdf:
                # results = [ [line...], [line...] ]
                for i, page_result in enumerate(results):
                    page_text = self._parse_single_page(page_result)
                    pages_content.append(page_text)
                    print(f"DEBUG OCR: Page {i+1} parsed, length: {len(page_text)}")
            else:
                # Single image
                page_text = self._parse_single_page(results)
                pages_content.append(page_text)

            full_text = "\n\n".join(pages_content)
            
            logger.info(f"OCR execution successful. Extracted {len(pages_content)} pages.")
            return {
                "source_file": str(file_path),
                "raw_text_content": full_text, # Legacy support
                "pages": pages_content,        # New page-by-page support
                "raw_data": results
            }

        except Exception as e:
            logger.exception("Unexpected error during OCR processing")
            raise OCRExecutionError(f"Failed to process {file_path}: {str(e)}") from e

    def _parse_single_page(self, page_result: Union[List, Any]) -> str:
        """
        Parse a single page's OCR result into a string.
        """
        text_lines = []
        if page_result is None:
            return ""
            
        try:
             # Check if it's an OCRResult object (PaddleOCR 3.x)
            if hasattr(page_result, 'rec_texts'):
                return "\n".join(page_result.rec_texts)
            elif isinstance(page_result, dict) and 'rec_texts' in page_result:
                return "\n".join(page_result['rec_texts'])
            
            # Legacy list format: [ [coords, [text, score]], ... ]
            if isinstance(page_result, list):
                for line in page_result:
                    if line and len(line) >= 2:
                        # line[1] is (text, score)
                        text, score = line[1]
                        text_lines.append(text)
        except Exception as e:
            logger.error(f"Error parsing page result: {e}")
            return str(page_result)
            
        return "\n".join(text_lines)

    def _parse_library_output(self, raw_results: Union[List, Any]) -> str:
        """Legacy helper, redirects to new logic if possible or deprecated."""
        # This is kept just in case but ideally shouldn't be used if we switch logic.
        # For simplicity, we won't fully implement it since process_document handles it.
        return ""

if __name__ == "__main__":
    # Internal Manual Test
    try:
        config.validate()
        print("Initializing PaddleOCR...")
        processor = PaddleOCRProcessor()
        print("PaddleOCRProcessor initialized via Library. Ready to process.")
    except Exception as e:
        print(f"Initialization failed: {e}")
