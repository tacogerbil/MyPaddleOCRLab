import logging
import argparse
from pathlib import Path
from typing import List
from config import config
from paddle_ocr import PaddleOCRProcessor, OCRExecutionError
from llm_cleanup import LLMCorrector, LLMConnectionError
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
    def process_file(self, file_path: Path, to_stdout: bool = False):
        """Run the full pipeline on a single file."""
        # Smart Path Resolution
        # If not absolute, try leveraging INPUT_DIR or just assume it's relative to CWD
        if not file_path.is_absolute():
            # If default logic, try relative to input dir. If user passed CLI path, assume CWD relative.
            # But here we handle both:
            possible_path = config.INPUT_DIR / file_path
            if possible_path.exists():
                file_path = possible_path
            # Else assume CWD relative (already covers file_path itself)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
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

            # 2. LLM Step
            if not to_stdout:
                logger.info(f"Step 2: Cleaning text with LLM...")
            
            cleaned_text = self.llm_corrector.cleanup_text(raw_text)
            
            # 3. Output
            if to_stdout:
                print(cleaned_text)
            else:
                output_file = config.CLEANED_DIR / f"{file_path.stem}_cleaned.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                logger.info(f"Success! Cleaned text saved to: {output_file}")

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
    
    args = parser.parse_args()
    
    orchestrator = WorkflowOrchestrator(llm_url=args.llm_url, llm_api_key=args.llm_api_key, llm_model=args.llm_model)
    
    if args.file:
        orchestrator.process_file(args.file, to_stdout=args.stdout)
    elif args.batch:
        orchestrator.run_batch(config.INPUT_DIR)
    else:
        # Default behavior if no args: check config INPUT_DIR
        print("No arguments provided. processing default INPUT_DIR...")
        orchestrator.run_batch(config.INPUT_DIR)
