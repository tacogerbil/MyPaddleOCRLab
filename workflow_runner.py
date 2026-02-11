import json
import logging
import argparse
import sys
import time
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
        logging.StreamHandler(sys.stderr)
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

    def _write_progress(self, dest_dir: Path, file_stem: str, page_num: int, total_pages: int, status: str, start_time: float):
        """Write progress.json for n8n or other tools to poll."""
        elapsed = time.time() - start_time
        avg_per_page = elapsed / page_num if page_num > 0 else 0
        remaining = avg_per_page * (total_pages - page_num)

        progress = {
            "file": file_stem,
            "page": page_num,
            "total_pages": total_pages,
            "percent": round((page_num / total_pages) * 100, 1) if total_pages > 0 else 0,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(remaining, 1),
        }
        progress_path = dest_dir / "progress.json"
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)

    def process_file(self, file_path: Path, output_dir: Path = None, to_stdout: bool = False, skip_llm: bool = False):
        """
        Run the full pipeline on a single file.

        When to_stdout=True: prints ONLY cleaned text to stdout (for n8n capture). No file write.
        When to_stdout=False: writes cleaned text to file. No stdout output.

        Resumable: skips pages whose output files already exist.
        """
        logger.info(f"ENTRY: process_file called with file_path={file_path}, to_stdout={to_stdout}, skip_llm={skip_llm}")

        # Smart Path Resolution
        if not file_path.is_absolute():
            logger.info(f"Path is relative, resolving...")
            possible_path = config.INPUT_DIR / file_path
            if possible_path.exists():
                file_path = possible_path

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        if file_path.is_dir():
            logger.error(f"Provided path '{file_path}' is a directory, not a file. Skipping.")
            return

        try:
            # 1. OCR Step
            logger.info(f"Step 1: Running OCR on {file_path.name}...")

            ocr_result = self.ocr_processor.process_document(file_path)
            raw_text = ocr_result.get("raw_text_content", "")

            if not raw_text.strip():
                logger.warning("OCR returned empty text. Skipping LLM cleanup.")
                return

            # 2. LLM Step (Page-by-Page)
            pages = ocr_result.get("pages", [])

            if not pages:
                 # Fallback for empty/legacy
                 pages = [ocr_result.get("raw_text_content", "")]

            total_pages = len(pages)
            logger.info(f"Processing {total_pages} page(s)")

            # Resolve output directory
            dest_dir = output_dir if output_dir else config.CLEANED_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename_suffix = "_pre_llm" if skip_llm else ""

            start_time = time.time()
            skipped = 0

            for i, page_text in enumerate(pages):
                page_num = i + 1
                page_filename = f"{file_path.stem}_page_{page_num:03d}{filename_suffix}.txt"
                output_path = dest_dir / page_filename

                # Resumability: skip pages that already have output files
                if not to_stdout and output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"  - Skipping Page {page_num}/{total_pages} (already exists: {page_filename})")
                    skipped += 1
                    continue

                if skip_llm:
                    cleaned_page = page_text
                else:
                    logger.info(f"  - Cleaning Page {page_num}/{total_pages} ({len(page_text)} chars)...")
                    try:
                        cleaned_page = self.llm_corrector.cleanup_text(page_text)
                    except Exception as e:
                        logger.error(f"Failed to clean page {page_num}: {e}")
                        cleaned_page = page_text

                if to_stdout:
                    print(cleaned_page)
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_page)
                    logger.info(f"  - Saved: {output_path}")
                    self._write_progress(dest_dir, file_path.stem, page_num, total_pages, "processing", start_time)

            # Write final progress
            if not to_stdout:
                self._write_progress(dest_dir, file_path.stem, total_pages, total_pages, "completed", start_time)

            if skipped > 0:
                logger.info(f"Resumed: skipped {skipped} already-completed pages.")
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
    parser.add_argument("--stdout", action="store_true", help="Print cleaned text to stdout only (for n8n). No file write.")
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
        print("No arguments provided. processing default INPUT_DIR...")
        orchestrator.run_batch(config.INPUT_DIR)
