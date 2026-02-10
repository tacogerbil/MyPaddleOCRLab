import logging
import argparse
from pathlib import Path
from typing import List
from config import config
from paddle_ocr import PaddleOCRProcessor, OCRExecutionError
from llm_cleanup import LLMCorrector, LLMConnectionError
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Configure Logging - MUST happen before any imports that use logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Force reconfiguration of root logger to ensure all modules use our handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "workflow.log"),
        logging.StreamHandler()
    ],
    force=True  # Override any previous configuration
)
logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """
    Coordinator class that stitches together OCR and LLM Cleanup.
    This replaces n8n for manual/local execution.
    """
    def __init__(self, llm_url: str = None, llm_api_key: str = None, llm_model: str = None):
        config.validate()
        self.ocr_processor = PaddleOCRProcessor()
        self.llm_corrector = LLMCorrector(llm_url=llm_url, llm_api_key=llm_api_key, llm_model=llm_model)
        
    def process_file(self, file_path: Path, output_dir: Path = None, to_stdout: bool = False, skip_llm: bool = False):
        """Run the full pipeline on a single file."""
        logger.info(f"ENTRY: process_file called with file_path={file_path}, to_stdout={to_stdout}")
        
        # Smart Path Resolution
        # If not absolute, try leveraging INPUT_DIR or just assume it's relative to CWD
        if not file_path.is_absolute():
            logger.info(f"Path is relative, resolving...")
            # If default logic, try relative to input dir. If user passed CLI path, assume CWD relative.
            # But here we handle both:
            possible_path = config.INPUT_DIR / file_path
            if possible_path.exists():
                file_path = possible_path
            # Else assume CWD relative (already covers file_path itself)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        if file_path.is_dir():
            if to_stdout:
                 print(f"ERROR: Provided path '{file_path}' is a directory, not a file. Did you mean to use --batch?")
            else:
                 logger.error(f"Provided path '{file_path}' is a directory, not a file. Skipping.")
            return

        try:
            # 1. OCR Step
            if not to_stdout:
                logger.info(f"Step 1: Running OCR on {file_path.name}...")
            else:
                # Debug output for n8n
                print(f"DEBUG: Processing file path: {file_path}")
                print(f"DEBUG: File exists: {file_path.exists()}")
                print(f"DEBUG: Absolute path: {file_path.absolute()}")
            
            ocr_result = self.ocr_processor.process_document(file_path)
            raw_text = ocr_result.get("raw_text_content", "")
            
            # Debug: Show what OCR returned
            if to_stdout:
                print(f"DEBUG: OCR returned {len(raw_text)} characters")
                print(f"DEBUG: First 100 chars: {raw_text[:100]}")
            
            if not raw_text.strip():
                if to_stdout:
                    print("ERROR: OCR returned empty text. No text detected in image.")
                else:
                    logger.warning("OCR returned empty text. Skipping LLM cleanup.")
                return

            # 2. LLM Step (Page-by-Page)
            pages = ocr_result.get("pages", [])
            cleaned_pages = []
            
            if not pages:
                 # Fallback for empty/legacy
                 pages = [ocr_result.get("raw_text_content", "")]
            
            # Determine Output Path
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{file_path.stem}.txt"
            else:
                output_path = config.CLEANED_DIR / f"{file_path.stem}.txt"
            
            # Reset file (overwrite) at start - ALWAYS create output file
            output_dir.mkdir(parents=True, exist_ok=True) if output_dir else config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")  # Clear file
            logger.info(f"Streaming output to: {output_path}")

            for i, page_text in enumerate(pages):
                page_num = i + 1
                
                # ... (rest of loop logic for cleaning) ...
                if skip_llm:
                    cleaned_page = page_text
                    if to_stdout:
                        print(f"DEBUG: Skipping LLM for Page {page_num}")
                else:
                    if not to_stdout:
                        logger.info(f"  - Cleaning Page {page_num}/{total_pages} ({len(page_text)} chars)...")
                    try:
                        cleaned_page = self.llm_corrector.cleanup_text(page_text)
                    except Exception as e:
                        logger.error(f"Failed to clean page {page_num}: {e}")
                        cleaned_page = page_text # Fallback

                # Output immediately
                separator = "\n\n--- PAGE BREAK ---\n\n" if i > 0 else ""
                content_to_write = separator + cleaned_page
                
                # Output immediately - write to BOTH stdout and file
                if to_stdout:
                    print(content_to_write)
                # Always write to file
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(content_to_write)
            
            if not to_stdout:
                logger.info(f"Success! Finished processing {total_pages} pages.")

        except (OCRExecutionError, LLMConnectionError) as e:
            logger.error(f"Workflow failed for {file_path.name}: {e}")
        except Exception as e:
            logger.exception(f"Critical workflow error for {file_path.name}")

    def run_batch(self, input_dir: Path):
        """Process all images/PDFs in the input directory."""
        files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        
        if not files:
            logger.info(f"No compatible files found in {input_dir}")
            return

        logger.info(f"Found {len(files)} files to process.")
        for file_path in files:
            self.process_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual OCR + LLM Workflow Runner")
    parser.add_argument("--file", type=Path, help="Specific file to process")
    parser.add_argument("--batch", action="store_true", help="Process all files in INPUT_DIR")
    parser.add_argument("--stdout", action="store_true", help="Print cleaned text to stdout (for n8n)")
    parser.add_argument("--llm-url", type=str, help="LLM API endpoint URL (overrides .env)")
    parser.add_argument("--llm-api-key", type=str, help="API Key for OpenWebUI/OpenAI (overrides .env)")
    parser.add_argument("--llm-model", type=str, help="LLM model name (overrides .env)")
    parser.add_argument("--output-dir", type=Path, help="Directory to save output files")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM cleanup, output raw OCR text")
    
    args = parser.parse_args()
    
    orchestrator = WorkflowOrchestrator(llm_url=args.llm_url, llm_api_key=args.llm_api_key, llm_model=args.llm_model)
    
    if args.file:
        orchestrator.process_file(args.file, output_dir=args.output_dir, to_stdout=args.stdout, skip_llm=args.skip_llm)
    elif args.batch:
        orchestrator.run_batch(config.INPUT_DIR)
    else:
        # Default behavior if no args: check config INPUT_DIR
        print("No arguments provided. processing default INPUT_DIR...")
        orchestrator.run_batch(config.INPUT_DIR)
