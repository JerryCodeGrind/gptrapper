from fastapi import FastAPI, File, Form, UploadFile
from typing import List

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str = Form(...), clips: List[UploadFile] = File(...)):
    print("Prompt:", prompt)
    print("Clips received:", len(clips))
    for i, clip in enumerate(clips):
        content = await clip.read()
        print(f"  clip_{i}: {clip.filename}, {len(content)} bytes")
    return {"status": "ok", "clips_received": len(clips)}
