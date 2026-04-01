import os
import time
import logging
from pathlib import Path
from typing import List, Tuple

from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-image")
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "prompts.txt")
OUT_DIR = Path(os.getenv("OUT_DIR", "output"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "3"))

API_KEY_1 = os.getenv("GEMINI_API_KEY_1", "").strip()
API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "").strip()


def load_prompts(file_path: str) -> List[str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        prompts.append(line)

    return prompts


def make_clients() -> List[genai.Client]:
    keys = [k for k in [API_KEY_1, API_KEY_2] if k]
    if not keys:
        raise ValueError("At least one Gemini API key is required in secrets.")

    return [genai.Client(api_key=key) for key in keys]


def split_half(prompts: List[str]) -> Tuple[List[str], List[str]]:
    mid = (len(prompts) + 1) // 2
    return prompts[:mid], prompts[mid:]


def save_first_image_from_response(response, filename_base: str) -> List[str]:
    saved_files = []

    # Save only the first image part for clean 0001.png style naming
    image_index = 1
    for part in getattr(response, "parts", []):
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue

        img = part.as_image()
        out_path = OUT_DIR / f"{filename_base}.png" if image_index == 1 else OUT_DIR / f"{filename_base}_{image_index}.png"
        img.save(out_path)
        saved_files.append(str(out_path))
        break

    return saved_files


def generate_for_prompt(client: genai.Client, prompt: str, filename_base: str) -> List[str]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
    )
    return save_first_image_from_response(response, filename_base)


def process_prompt_with_retry(
    clients: List[genai.Client],
    prompt: str,
    index: int,
    assigned_client_index: int
) -> bool:
    filename_base = f"{index:04d}"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        # first retry uses assigned client, next retry flips to the other client
        client_index = (assigned_client_index + (attempt - 1)) % len(clients)
        client = clients[client_index]

        try:
            logging.info(
                f"Prompt {index:04d} | attempt {attempt}/{MAX_RETRIES} | client {client_index + 1}"
            )
            saved = generate_for_prompt(client, prompt, filename_base)

            if not saved:
                raise RuntimeError("Model returned no image data.")

            logging.info(f"Saved: {saved[0]}")
            return True

        except Exception as e:
            last_error = e
            logging.warning(f"Prompt {index:04d} failed on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    logging.error(f"Prompt {index:04d} failed permanently: {last_error}")
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(PROMPTS_FILE)
    if not prompts:
        raise ValueError("No prompts found in prompts.txt")

    clients = make_clients()

    # Half-half split between the keys
    half_1, half_2 = split_half(prompts)

    jobs = []
    for i, prompt in enumerate(half_1, start=1):
        jobs.append((prompt, i, 0))  # key 1
    for i, prompt in enumerate(half_2, start=len(half_1) + 1):
        jobs.append((prompt, i, min(1, len(clients) - 1)))  # key 2 if present

    success_count = 0
    for prompt, idx, client_idx in jobs:
        ok = process_prompt_with_retry(clients, prompt, idx, client_idx)
        if ok:
            success_count += 1

    logging.info(f"Done. Success: {success_count}/{len(prompts)}")


if __name__ == "__main__":
    main()
