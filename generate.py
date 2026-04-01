#!/usr/bin/env python3
"""
Gemini Image Generator
2 API keys = ~1000 images/day free
Images saved as: 1.png, 2.png, 3.png...
"""

import google.generativeai as genai
import base64, os, time, json
from pathlib import Path

# ── Config ───────────────────────────────────────────────
KEY_ID  = int(os.environ.get("KEY_ID", "1"))
API_KEY = os.environ.get(f"GEMINI_KEY_{KEY_ID}", "")

if not API_KEY:
    print(f"ERROR: Secret GEMINI_KEY_{KEY_ID} not set in GitHub!")
    exit(1)

# ── Load prompts.txt ─────────────────────────────────────
src = Path("prompts.txt")
if not src.exists():
    print("ERROR: prompts.txt not found in repo!")
    exit(1)

ALL = [l.strip() for l in src.read_text("utf-8").splitlines()
       if l.strip() and not l.startswith("#")]
TOTAL = len(ALL)

# Key 1 = prompts 1,3,5,7... (index 0,2,4,6...)
# Key 2 = prompts 2,4,6,8... (index 1,3,5,7...)
mine = [(i, ALL[i]) for i in range(TOTAL) if i % 2 == (KEY_ID - 1)]

print(f"Key {KEY_ID} | {len(mine)} prompts | model: gemini-2.0-flash-exp")

# ── Setup ────────────────────────────────────────────────
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ── Generate ─────────────────────────────────────────────
def make(prompt, num):
    for attempt in range(1, 4):
        try:
            resp = model.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    response_modalities=["IMAGE"]
                )
            )
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = base64.b64decode(part.inline_data.data)
                    if len(data) > 5000:
                        return data
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                print(f"  rate limit — wait 65s")
                time.sleep(65)
            elif "400" in err:
                print(f"  bad request, skip")
                return None
            else:
                print(f"  error: {err[:50]}, retry {attempt}/3")
                time.sleep(attempt * 5)
    return None

ok = fail = 0
for idx, prompt in mine:
    n    = idx + 1
    path = OUT / f"{n}.png"

    if path.exists() and path.stat().st_size > 5000:
        print(f"skip {n}.png")
        ok += 1
        continue

    print(f"gen {n}/{TOTAL}: {prompt[:55]}...")
    data = make(prompt, n)

    if data:
        path.write_bytes(data)
        print(f"  saved {n}.png ({len(data)//1024}KB)")
        ok += 1
    else:
        print(f"  FAILED {n}.png")
        fail += 1

    time.sleep(7)  # 10 RPM limit = min 6s between requests

print(f"\nKey {KEY_ID}: {ok} ok, {fail} failed")
(OUT / f"report_{KEY_ID}.json").write_text(
    json.dumps({"key": KEY_ID, "ok": ok, "fail": fail})
)

