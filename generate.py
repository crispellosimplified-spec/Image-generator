#!/usr/bin/env python3
# Confirmed working - gemini-2.0-flash-exp
# Source: developers.googleblog.com + dev.to verified examples

from google import genai
from google.genai import types
import os, time, json
from pathlib import Path

KEY_ID  = int(os.environ.get("KEY_ID", "1"))
API_KEY = os.environ.get(f"GEMINI_KEY_{KEY_ID}", "")

if not API_KEY:
    print(f"ERROR: GEMINI_KEY_{KEY_ID} secret not set!")
    exit(1)

src = Path("prompts.txt")
if not src.exists():
    print("ERROR: prompts.txt not found!")
    exit(1)

ALL   = [l.strip() for l in src.read_text("utf-8").splitlines()
         if l.strip() and not l.startswith("#")]
TOTAL = len(ALL)
mine  = [(i, ALL[i]) for i in range(TOTAL) if i % 2 == (KEY_ID - 1)]

print(f"Key {KEY_ID} | {len(mine)} prompts | gemini-2.0-flash-exp")

client = genai.Client(api_key=API_KEY)
OUT    = Path("output")
OUT.mkdir(exist_ok=True)

def make(prompt):
    for attempt in range(1, 4):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            )
            for part in resp.candidates[0].content.parts:
                if part.inline_data is not None:
                    d = part.inline_data.data
                    if len(d) > 1000:
                        return d
            print(f"  no image returned, attempt {attempt}")
        except Exception as e:
            err = str(e)
            print(f"  err attempt {attempt}: {err[:80]}")
            if "429" in err or "quota" in err.lower():
                print("  rate limit - sleep 65s")
                time.sleep(65)
            elif "400" in err:
                return None
            else:
                time.sleep(attempt * 4)
    return None

ok = fail = 0
for idx, prompt in mine:
    n    = idx + 1
    path = OUT / f"{n}.png"
    if path.exists() and path.stat().st_size > 1000:
        print(f"skip {n}.png"); ok += 1; continue

    print(f"gen {n}/{TOTAL}: {prompt[:55]}...")
    data = make(prompt)
    if data:
        path.write_bytes(data)
        print(f"  OK {n}.png ({len(data)//1024}KB)"); ok += 1
    else:
        print(f"  FAIL {n}.png"); fail += 1
    time.sleep(6)

print(f"\nKey {KEY_ID}: {ok} ok, {fail} fail")
(OUT / "report.json").write_text(json.dumps({"key": KEY_ID, "ok": ok, "fail": fail}))
