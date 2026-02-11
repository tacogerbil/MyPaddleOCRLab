import gc
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Disable PaddleX check for connectivity to model hosters (User Request)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# We import PaddleOCR inside the class or method to avoid hard dependency if library is missing during development/mocking
# But for production code, it should be at top level.
# Core dependencies
try:
    from paddleocr import PaddleOCR
    import numpy as np
    from pdf2image import convert_from_path, pdfinfo_from_path
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None # Disable decompression bomb check
except ImportError:
    PaddleOCR = None
    np = None
    convert_from_path = None
    PIL = None

# Optional Layout Analysis dependency
try:
    from paddleocr import PPStructureV3
except ImportError:
    PPStructureV3 = None

# Optional dependencies
try:
    import cv2
except ImportError:
    cv2 = None

from config import config

# Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # disable global config
logger = logging.getLogger(__name__)

class OCRExecutionError(Exception):
    """Custom exception for errors during OCR execution."""
    pass

class PaddleOCRProcessor:
    """
    Adapter class for PaddleOCR (Python Library Version).
    Responsible for executing the OCR engine on a given image/PDF and returning the result.
    
    Implements Singleton pattern for the heavy model loading.
    
    Supports:
    1. Standard OCR (Text Detection + Recognition)
    2. Layout Analysis (PPStructure) for header/footer filtering
    """
    _engine_instance = None
    _structure_instance = None

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
            
            # Initialize core OCR engine (must succeed)
            try:
                device_arg = "gpu" if config.USE_GPU else "cpu"
                cls._engine_instance = PaddleOCR(
                    use_angle_cls=config.USE_ANGLE_CLS,
                    lang=config.LANG,
                    device=device_arg,
                    det_limit_side_len=config.DET_LIMIT_SIDE_LEN,
                    det_limit_type=config.DET_LIMIT_TYPE,
                    rec_batch_num=config.REC_BATCH_NUM
                )
                logger.info("PaddleOCR Engine initialized successfully.")
            except Exception as e:
                raise OCRExecutionError(f"PaddleOCR init failed: {e}") from e

            # Initialize Layout Analysis (must succeed if enabled)
            if config.ENABLE_LAYOUT_ANALYSIS:
                if PPStructureV3 is None:
                    raise OCRExecutionError(
                        "ENABLE_LAYOUT_ANALYSIS=true but PPStructureV3 not importable. "
                        "Install with: pip install 'paddlex[ocr]'")
                try:
                    logger.info("Initializing PPStructureV3 (Layout Analysis)...")
                    cls._structure_instance = PPStructureV3(
                        use_doc_orientation_classify=config.USE_ANGLE_CLS,
                        lang=config.LANG
                    )
                    logger.info("PPStructureV3 initialized successfully.")
                except Exception as e:
                    raise OCRExecutionError(
                        f"PPStructureV3 init failed: {e}. "
                        "Install deps with: pip install 'paddlex[ocr]'") from e

    def _parse_structure_v3_output(self, structure_result: Any, img_height: int) -> str:
        """
        Parse PPStructureV3 output and filter unwanted regions.
        
        PPStructureV3 returns layout regions with types like:
        - 'text': Body text
        - 'title': Section/story titles
        - 'header': Page headers
        - 'footer': Page footers  
        - 'figure': Images
        - 'figure_caption': Image captions
        - 'table': Tables
        
        We want to keep: text, title, table
        We want to remove: header, footer, figure, figure_caption
        """
        try:
            # PPStructureV3.predict() returns a list of page results
            # Each page result has layout_det_res and overall_ocr_res
            if not structure_result:
                return ""
            
            # Get the first (and only) page result
            page_result = structure_result[0] if isinstance(structure_result, list) else structure_result
            
            # Extract layout detection results
            layout_regions = page_result.layout_det_res if hasattr(page_result, 'layout_det_res') else []
            
            # Filter and extract text from valid regions
            valid_text_blocks = []
            
            for region in layout_regions:
                region_type = region.get('type', '').lower()
                
                # Skip unwanted region types
                if region_type in ['header', 'footer', 'figure', 'figure_caption']:
                    logger.debug(f"Filtering out {region_type} region")
                    continue
                
                # Extract text from this region
                region_text = region.get('text', '')
                if region_text and region_text.strip():
                    valid_text_blocks.append(region_text.strip())
            
            # Join all valid text blocks
            return '\n\n'.join(valid_text_blocks)
            
        except Exception as e:
            logger.error(f"Failed to parse PPStructureV3 output: {e}")
            # Fallback: try to extract any text we can
            try:
                if hasattr(structure_result, 'overall_ocr_res'):
                    return str(structure_result.overall_ocr_res)
                return str(structure_result)
            except:
                return ""

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Run OCR on the given document (Image or PDF).
        Processes PDFs page-by-page to avoid OOM on large books.
        Returns a dictionary containing metadata and extracted text.

        Note: For streaming (save-as-you-go), use iter_pages() instead.
        """
        pages_content = []
        for page_num, total_pages, page_text in self.iter_pages(file_path):
            pages_content.append(page_text)

        full_text = "\n\n".join(pages_content)
        logger.info(f"OCR execution successful. Extracted {len(pages_content)} pages.")
        return {
            "source_file": str(file_path),
            "raw_text_content": full_text,
            "pages": pages_content,
        }

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
            
            # Check if it's a list containing the result dict (Paddle v3/v4 behavior on some pipelines)
            # The log shows: [{'rec_texts': [...], ...}]
            if isinstance(page_result, list):
                for line in page_result:
                    # Modern format: dict inside list
                    if isinstance(line, dict) and 'rec_texts' in line:
                        text_lines.extend(line['rec_texts'])
                        continue

                    # Legacy list format: [ [coords], [text, score] ]
                    if isinstance(line, list) and len(line) >= 2:
                        try:
                            # line[1] is (text, score)
                            text_data = line[1]
                            if isinstance(text_data, (list, tuple)) and len(text_data) >= 1:
                                text_lines.append(text_data[0])
                        except Exception:
                            continue

        except Exception as e:
            logger.exception(f"Error parsing page result: {e}. Raw result: {page_result}")
            return str(page_result)
            
        return "\n".join(text_lines)

    def _ocr_single_image(self, img_input) -> str:
        """
        OCR a single image (numpy array or file path string).
        Uses PPStructureV3 if enabled, standard OCR otherwise.
        No silent fallbacks — if layout analysis fails, it raises so the problem is visible.
        """
        if config.ENABLE_LAYOUT_ANALYSIS and self._structure_instance:
            result = self._structure_instance.predict(input=img_input)
            img_height = img_input.shape[0] if hasattr(img_input, 'shape') else 0
            return self._parse_structure_v3_output(result, img_height)

        result = self._engine_instance.ocr(img_input if isinstance(img_input, str) else img_input)
        return self._parse_single_page(result)

    def iter_pages(self, file_path: Path):
        """
        Generator that yields (page_num, total_pages, page_text) one page at a time.
        For PDFs: converts and OCRs each page sequentially, yielding immediately.
        For images: yields a single page.
        Supports PPStructureV3 layout analysis when enabled.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        file_path = file_path.resolve()
        logger.info(f"Processing file: {file_path}")
        is_pdf = file_path.suffix.lower() == '.pdf'

        if is_pdf and convert_from_path:
            if not os.path.exists(str(file_path)):
                raise FileNotFoundError(f"File vanished before processing: {file_path}")

            info = pdfinfo_from_path(str(file_path))
            total_pages = info.get('Pages', 0)
            logger.info(f"PDF detected with {total_pages} pages. Streaming page-by-page.")

            for i in range(total_pages):
                page_num = i + 1
                logger.info(f"OCR: Processing Page {page_num}/{total_pages}...")

                images = convert_from_path(str(file_path), first_page=page_num, last_page=page_num, fmt='jpeg', dpi=config.PDF_DPI)

                if not images:
                    yield (page_num, total_pages, "")
                    continue

                img_array = np.array(images[0])
                page_text = self._ocr_single_image(img_array)

                del img_array
                del images
                gc.collect()

                yield (page_num, total_pages, page_text)

        elif is_pdf:
            logger.warning("pdf2image not found. Loading entire PDF into RAM...")
            results = self._engine_instance.ocr(str(file_path))
            total_pages = len(results) if results else 0
            for i, page_result in enumerate(results):
                yield (i + 1, total_pages, self._parse_single_page(page_result))
        else:
            # Single image
            logger.info(f"OCR: Processing image {file_path.name}...")
            page_text = self._ocr_single_image(str(file_path))
            yield (1, 1, page_text)

if __name__ == "__main__":
    # Internal Manual Test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config.validate()
        print("Initializing PaddleOCR...")
        processor = PaddleOCRProcessor()
        print("PaddleOCRProcessor initialized via Library. Ready to process.")
    except Exception as e:
        print(f"Initialization failed: {e}")
