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
    from paddleocr import PPStructure
except ImportError:
    PPStructure = None

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
            
            try:
                # PaddleOCR 3.x initialization with modern parameters
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

                # Initialize Layout Analysis if enabled
                if config.ENABLE_LAYOUT_ANALYSIS:
                    logger.info("Initializing PPStructure (Layout Analysis)...")
                    cls._structure_instance = PPStructure(
                        image_orientation=config.USE_ANGLE_CLS,
                        lang=config.LANG
                    )
                    logger.info("PPStructure initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise OCRExecutionError(f"Initialization failed: {e}")

    def _detect_horizontal_lines(self, img_array: Any) -> Dict[str, List[int]]:
        """
        Detect logical header/footer separator lines using morphological operations.
        Returns Y-coordinates of significant horizontal lines at top/bottom of page.
        """
        if cv2 is None:
            return {"top": [], "bottom": []}

        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array

            # Apply adaptive threshold to get binary image
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Create horizontal kernel structure
            # We want lines that are at least 40% of page width
            h, w = gray.shape
            min_width = int(w * 0.4)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))

            # Detect lines
            detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            top_lines = []
            bottom_lines = []

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > min_width:
                    # Determine logical position (0-1.0)
                    rel_y = y / h
                    # Top 15% is Header Candidate
                    if rel_y < 0.15:
                        top_lines.append(y + ch) # Bottom of the line
                    # Bottom 15% is Footer Candidate 
                    elif rel_y > 0.85:
                        bottom_lines.append(y) # Top of the line

            return {"top": top_lines, "bottom": bottom_lines}
            
        except Exception as e:
            logger.warning(f"Line detection failed: {e}")
            return {"top": [], "bottom": []}

    def _filter_regions(self, regions: List[Dict], img_height: int, lines: Dict[str, List[int]]) -> List[str]:
        """
        Filter layout regions based on position (headers/footers) and heuristics (captions).
        """
        if not regions:
            return []

        # 1. Determine safe zones from lines
        # Ignore anything ABOVE the lowest top line (header separator)
        header_limit = max(lines["top"]) if lines["top"] else 0
        
        # Ignore anything BELOW the highest bottom line (footer separator)
        footer_limit = min(lines["bottom"]) if lines["bottom"] else img_height

        valid_regions = []
        text_heights = []

        # 2. First Pass: Position & Type Filtering
        for region in regions:
            bbox = region['bbox'] # [x1, y1, x2, y2]
            y1, y2 = bbox[1], bbox[3]
            r_type = region['type']
            res = region.get('res', [])

            # Filter Figures/Images
            if r_type == 'figure':
                continue
                
            # Filter Headers (Above line)
            if y2 < header_limit:
                continue
                
            # Filter Footers (Below line)
            if y1 > footer_limit:
                continue

            # Check if it's explicitly classified as header/footer by model
            if r_type in ['header', 'footer']:
                continue

            # Calculate height for heuristic stats
            height = y2 - y1
            text_heights.append(height)
            
            valid_regions.append({
                'text': '\n'.join([line['text'] for line in res]),
                'y1': y1,
                'y2': y2,
                'height': height
            })

        # 3. Second Pass: Caption Heuristics (Size & Gap)
        # We need median height to detect "small" text
        if not text_heights:
            return []
            
        median_height = sorted(text_heights)[len(text_heights) // 2]
        final_text = []
        
        # Sort by Y position
        valid_regions.sort(key=lambda x: x['y1'])
        
        for i, region in enumerate(valid_regions):
            # Check for Bottom-of-Page Caption
            # If it's in the bottom 20%, verify size/gap
            is_bottom = region['y2'] > (img_height * 0.8)
            
            if is_bottom and config.IGNORE_CAPTIONS:
                # Heuristic A: Small Font (Footnote)
                if region['height'] < (median_height * 0.85):
                    continue
                
                # Heuristic B: Large Gap (Image Caption)
                # Check distance from previous region
                if i > 0:
                    prev_region = valid_regions[i-1]
                    gap = region['y1'] - prev_region['y2']
                    # If gap is > 5% of page height, it's likely a detached caption
                    if gap > (img_height * 0.05):
                        continue

            final_text.append(region['text'])

        return final_text

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Run OCR on the given document (Image or PDF).
        Returns a dictionary containing metadata and extracted text.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        file_path = file_path.resolve()
        is_pdf = file_path.suffix.lower() == '.pdf'
        pages_content = []
        raw_results = []
        
        try:
            if is_pdf:
                # Use pdf2image for robust rendering
                logger.info(f"Converting PDF to images: {file_path.name}")
                
                # Get total pages first
                # info = pdfinfo_from_path(file_path)
                # total_pages = info["Pages"]
                
                # Convert PDF to images in memory
                images = convert_from_path(str(file_path), dpi=config.PDF_DPI)
                
                logger.info(f"Processing {len(images)} pages...")
                
                for i, img in enumerate(images):
                    # Convert to numpy array for PaddleOCR
                    img_np = np.array(img)
                    
                    if config.ENABLE_LAYOUT_ANALYSIS and self._structure_instance:
                        # --- Layout Analysis Flow ---
                        # 1. Detect Lines
                        lines = self._detect_horizontal_lines(img_np)
                        
                        # 2. Run PPStructure
                        # Need to convert RGB (PIL default) to BGR for OpenCV/Paddle
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if cv2 else img_np
                        result = self._structure_instance(img_bgr)
                        
                        # 3. Filter & Extract
                        page_text = self._filter_regions(result, img_np.shape[0], lines)
                        pages_content.append("\n\n".join(page_text))
                        raw_results.append(result)
                        
                    else:
                        # --- Standard OCR Flow ---
                        result = self._engine_instance.ocr(img_np, cls=config.USE_ANGLE_CLS)
                        pages_content.append(self._parse_single_page(result))
                        raw_results.append(result)
                        
                    # Explicit cleanup
                    del img_np
                    if i % 5 == 0:
                        gc.collect()

            else:
                # Single Image
                if config.ENABLE_LAYOUT_ANALYSIS and self._structure_instance:
                    # Layout Analysis for Image
                    if cv2:
                        img = cv2.imread(str(file_path))
                        lines = self._detect_horizontal_lines(img)
                        result = self._structure_instance(img)
                        page_text = self._filter_regions(result, img.shape[0], lines)
                        pages_content.append("\n\n".join(page_text))
                    else:
                        # Fallback if CV2 missing
                        result = self._engine_instance.ocr(str(file_path), cls=config.USE_ANGLE_CLS)
                        pages_content.append(self._parse_single_page(result))
                else:
                    # Standard OCR
                    result = self._engine_instance.ocr(str(file_path), cls=config.USE_ANGLE_CLS)
                    pages_content.append(self._parse_single_page(result))
                
                raw_results.append(result)

            full_text = "\n\n".join(pages_content)
            
            logger.info(f"OCR execution successful. Extracted {len(pages_content)} pages.")
            return {
                "source_file": str(file_path),
                "raw_text_content": full_text, # Legacy support
                "pages": pages_content,        # New page-by-page support
                "raw_data": raw_results
            }

        except Exception as e:
            logger.error(f"OCR Execution failed: {e}")
            raise OCRExecutionError(f"Processing failed: {e}")

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

    def iter_pages(self, file_path: Path):
        """
        Generator that yields (page_num, total_pages, page_text) one page at a time.
        For PDFs: converts and OCRs each page sequentially, yielding immediately.
        For images: yields a single page.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

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
                result = self._engine_instance.ocr(img_array)
                page_text = self._parse_single_page(result)

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
            results = self._engine_instance.ocr(str(file_path))
            page_text = self._parse_single_page(results)
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
