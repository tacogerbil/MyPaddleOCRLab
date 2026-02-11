import requests
import logging
import time
from enum import Enum
from typing import Optional, Dict, Any
from config import config

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPEN_WEBUI = "open_webui"

class LLMConnectionError(Exception):
    """Custom exception for errors connecting to the LLM API."""
    pass

class LLMCorrector:
    """
    Adapter class for Local LLM interaction.
    Responsible for sending dirty OCR text to the LLM and retrieving the cleaned version.
    """

    MAX_RETRIES = 1
    RETRY_BACKOFF = 5  # seconds

    def __init__(self, llm_url: str = None, llm_api_key: str = None, llm_model: str = None):
        """Initialize using global config or provided overrides."""
        self.api_url = llm_url if llm_url else config.LLM_API_URL
        self.api_key = llm_api_key
        self.model_name = llm_model if llm_model else config.LLM_MODEL_NAME
        self.timeout = config.LLM_TIMEOUT

        # Auto-fix OpenWebUI URL once at init, not on every request
        if self.api_key and not self.api_url.endswith("/chat/completions"):
            base = self.api_url.rstrip("/")
            if "api" not in base and "v1" not in base:
                self.api_url = f"{base}/api/chat/completions"
                logger.info(f"Auto-corrected LLM URL to: {self.api_url}")

    def cleanup_text(self, raw_text: str) -> str:
        """
        Send raw OCR text to the LLM for correction.
        Retries once on transient failures before falling back to raw text.
        """
        if not raw_text.strip():
            return ""

        last_error = None
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.warning(f"LLM retry {attempt}/{self.MAX_RETRIES} after {self.RETRY_BACKOFF}s backoff...")
                    time.sleep(self.RETRY_BACKOFF)
                response = self._send_request(raw_text)
                return self._extract_content(response)
            except LLMConnectionError as e:
                last_error = e
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM attempt {attempt + 1} unexpected error: {e}")

        logger.error(f"LLM cleanup failed after {1 + self.MAX_RETRIES} attempts. Falling back to raw text.")
        return raw_text

    def _send_request(self, prompt: str) -> Dict[str, Any]:
        """
        Send HTTP POST request to the LLM API.
        Supports both Ollama (/api/generate) and OpenWebUI/OpenAI (/v1/chat/completions) formats.
        """
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        is_chat_api = "chat/completions" in self.api_url or self.api_key is not None

        if is_chat_api:
             payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": """You are a strict OCR text corrector. Your task is to output corrected plain text only.
Do not summarize, rephrase, paraphrase, reorganize, or add content.
Do not be creative.
The output must be identical to the input except for fixing clear OCR errors, spelling errors and broken line issues defined below.

Allowed corrections:

Character-level OCR fixes (context required):
- Fix common OCR confusions only when context makes the correction obvious:
  - Numeric context: O/o→0, l/I→1, S→5, B→8, Z→2
  - Word context: 0→O/o, 1→l, 5→S only if it forms a valid word
- Fix ligatures and OCR artifacts (ﬁ→fi, ﬂ→fl, stray symbols)
- If ambiguous, do not change

Spelling Errors:
- Correct obvious spelling mistakes

Dot-leader + page number line repair (critical):
- If a line appears to be a table of contents entry (title text)
- AND the following line contains only:
  - dots + a number (e.g. .2, ..ix, ....17)
  - or a number/roman numeral alone
- THEN merge them into a single line:
  - Preserve the dots
  - Add exactly one space before the page number
- Examples of lines to merge:
  - Title\\n.2
  - Title\\n..ix
  - Title .....\\n17

Line-break fixes:
- Remove line breaks that split:
  - a word
  - a dot-leader/page-number pair
- Do not remove real paragraph breaks
- If unsure, keep the line break

Output rules:
- Output text only (no markup, no explanations)
- Preserve original wording, order, casing, and punctuation except for OCR fixes
- Make the minimum number of changes required
- Be conservative. Only fix errors that are clearly caused by OCR."""},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }

        logger.info(f"LLM request to {self.api_url} (model={self.model_name}, input={len(prompt)} chars, timeout={self.timeout}s)")

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)

            if response.status_code == 404:
                 logger.error(f"LLM 404 Error. URL used: {self.api_url}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"HTTP Request failed: {e}")

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        """Parse strict response from JSON."""
        if "response" in response_json:
            return response_json["response"]

        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0].get("message", {}).get("content", "")

        logger.error(f"Unknown JSON response format: {response_json.keys()}")
        raise LLMConnectionError("Could not parse LLM response.")

if __name__ == "__main__":
    try:
        corrector = LLMCorrector()
        print("LLMCorrector initialized.")
    except Exception as e:
        print(f"Init failed: {e}")
