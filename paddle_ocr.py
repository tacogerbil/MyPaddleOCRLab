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
        
        Raises:
            OCRExecutionError: If execution fails.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        
        try:
            # Run OCR (angle classification controlled via use_angle_cls during init)
            print(f"DEBUG OCR: Calling PaddleOCR.ocr() on {file_path}...")
            results = self._engine_instance.ocr(str(file_path))
            
            # DEBUG: Inspect raw results
            print(f"DEBUG OCR: Call completed. Result type: {type(results)}")
            print(f"DEBUG OCR: Result length: {len(results) if results else 0}")
            if results:
                print(f"DEBUG OCR: First element type: {type(results[0])}")
                print(f"DEBUG OCR: First element content: {results[0]}")
            
            if not results:
                logger.warning(f"No text detected in {file_path}")
                return {"source_file": str(file_path), "raw_text_content": ""}

            # Parse and normalize
            parsed_text = self._parse_library_output(results)
            print(f"DEBUG OCR: Parsed text length: {len(parsed_text)}")
            
            logger.info("OCR execution successful.")
            return {
                "source_file": str(file_path),
                "raw_text_content": parsed_text,
                "raw_data": results # Keep raw data for debugging/advanced usage if needed
            }

        except Exception as e:
            logger.exception("Unexpected error during OCR processing")
            raise OCRExecutionError(f"Failed to process {file_path}: {str(e)}") from e

    def _parse_library_output(self, raw_results: Union[List, Any]) -> str:
        """
        Convert PaddleOCR's complex structure into a single string.
        Handles both OCRResult objects (PaddleOCR 3.x) and legacy list formats.
        """
        text_lines = []
        
        if raw_results is None:
            return ""
        
        # PaddleOCR 3.x returns a list of OCRResult objects
        # Each OCRResult has a 'rec_texts' field containing the detected text
        try:
            for result in raw_results:
                # Check if it's an OCRResult object (has rec_texts attribute)
                if hasattr(result, 'rec_texts'):
                    text_lines.extend(result.rec_texts)
                # Or if it's a dict with 'rec_texts' key
                elif isinstance(result, dict) and 'rec_texts' in result:
                    text_lines.extend(result['rec_texts'])
                # Legacy format: list of lines
                elif isinstance(result, list):
                    for line in result:
                        if line and len(line) >= 2:
                            # Format: [coords, (text, score)]
                            text, score = line[1]
                            text_lines.append(text)
        
        except Exception as e:
            logger.error(f"Error parsing PaddleOCR output structure: {e}")
            # Fallback string conversion for debugging
            return str(raw_results)

        return "\n".join(text_lines)

if __name__ == "__main__":
    # Internal Manual Test
    try:
        config.validate()
        print("Initializing PaddleOCR...")
        processor = PaddleOCRProcessor()
        print("PaddleOCRProcessor initialized via Library. Ready to process.")
    except Exception as e:
        print(f"Initialization failed: {e}")
