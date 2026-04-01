#!/usr/bin/env python3
from google import genai
from google.genai import types
import os, time, json
from pathlib import Path

KEY_ID  = int(os.environ.get("KEY_ID", "1"))
API_KEY = os.environ.get(f"GEMINI_KEY_{KEY_ID}", "")

if not API_KEY:
    print(f"ERROR: GEMINI_KEY_{KEY_ID} secret missing!")
    exit(1)

src = Path("prompts.txt")
if not src.exists():
    print("ERROR: prompts.txt missing!")
    exit(1)

ALL   = [l.strip() for l in src.read_text("utf-8").splitlines()
         if l.strip() and not l.startswith("#")]
TOTAL = len(ALL)
mine  = [(i, ALL[i]) for i in range(TOTAL) if i % 2 == (KEY_ID - 1)]

print(f"Key{KEY_ID} | {len(mine)} prompts | model: gemini-2.5-flash-image-preview")

client = genai.Client(api_key=API_KEY)
OUT    = Path("output")
OUT.mkdir(exist_ok=True)

def make(prompt):
    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )
        for part in r.candidates[0].content.parts:
            if part.inline_data is not None:
                d = part.inline_data.data
                if len(d) > 1000:
                    return d
        return None
    except Exception as e:
        print(f"  ERR: {str(e)[:80]}")
        if "429" in str(e):
            time.sleep(60)
        return None

ok = fail = 0
for idx, prompt in mine:
    n    = idx + 1
    path = OUT / f"{n}.png"
    if path.exists() and path.stat().st_size > 1000:
        print(f"skip {n}.png"); ok += 1; continue

    print(f"gen {n}/{TOTAL}...", flush=True)
    data = make(prompt)
    if data:
        path.write_bytes(data)
        print(f"  OK {n}.png {len(data)//1024}KB"); ok += 1
    else:
        print(f"  FAIL {n}.png"); fail += 1
    time.sleep(6)

print(f"\nDone: {ok} ok, {fail} fail")
(OUT/"report.json").write_text(json.dumps({"ok":ok,"fail":fail}))
