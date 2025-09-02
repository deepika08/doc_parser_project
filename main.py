import os
import json
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
import PyPDF2
import docx

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Env ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- App + storage ----
app = FastAPI(title="LangChain Compliance Agent (Fixed)")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc"
}

# ---- Text extraction helpers ----
def extract_text_from_pdf(file_path: Path) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

def extract_text_from_docx(file_path: Path) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()

# ---- Helper: robust JSON extraction from model output ----
def extract_json_from_text(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # try to find first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = s[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
    return None

# ---- LangChain wrapper ----
def analyze_with_langchain(text: str, guidelines: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    # short-circuit extremely long docs (you could instead chunk + aggregate)
    text_snippet = text if len(text) < 6000 else text[:6000] + "\n\n[TRUNCATED]"

    # build a strict prompt that asks for JSON only
    prompt_template = """You are an expert editor and compliance checker for English writing.
Analyze the DOCUMENT (below) against the GUIDELINES (below) and RETURN A VALID JSON OBJECT ONLY (no explanation) with these keys:
- summary: {{ "compliant": bool, "message": str }}
- violations: [ {{ "rule": str, "message": str, "examples": [str] }} ... ]
- suggestions: [str]
- metrics: {{ "word_count": int, "sentence_count": int, "readability_note": str }}

GUIDELINES:
{guidelines}

DOCUMENT:
{text}
"""
    prompt = PromptTemplate(input_variables=["guidelines", "text"], template=prompt_template)

    # instantiate ChatOpenAI via LangChain
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)

    chain = LLMChain(llm=llm, prompt=prompt)

    # run. use predict (safe) and capture output
    logger.info("Calling LLM for analysis (text length=%d)", len(text_snippet))
    model_output = chain.predict(guidelines=guidelines, text=text_snippet)
    logger.debug("Model raw output: %s", model_output[:1000])

    parsed = extract_json_from_text(model_output)
    if parsed is None:
        # return raw for debugging but mark as non-compliant
        return {
            "summary": {"compliant": False, "message": "Model did not return parseable JSON"},
            "violations": [],
            "suggestions": [],
            "metrics": {},
            "raw_output": model_output
        }
    return parsed

# ---- UI & API endpoints ----
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <head><title>LangChain Compliance Checker</title></head>
      <body>
        <h2>Upload a document for compliance analysis</h2>
        <a href="/upload-form">Upload form</a>
      </body>
    </html>
    """

@app.get("/upload-form", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
      <head><title>Upload & Analyze</title></head>
      <body>
        <h2>Upload a PDF or Word Document</h2>
        <form action="/analyze/" enctype="multipart/form-data" method="post">
          <input type="file" name="file" accept=".pdf,.doc,.docx" required><br/><br/>
          <label>Guidelines (plain text):</label><br/>
          <textarea name="guidelines" rows="8" cols="80" placeholder="Avoid passive voice. Sentences under 25 words. Formal tone."></textarea><br/><br/>
          <button type="submit">Analyze</button>
        </form>
      </body>
    </html>
    """

@app.post("/analyze/")
async def analyze_upload(
    file: UploadFile = File(...),
    guidelines: Optional[str] = Form(default="Avoid passive voice. Sentences under 25 words. Formal tone.")
):
    # validate mime
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    ext = ALLOWED_TYPES[file.content_type]
    fname = f"{uuid.uuid4()}{ext}"
    fpath = UPLOAD_DIR / fname

    # save file
    try:
        with fpath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("File save error")
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")

    # extract text
    try:
        if ext == ".pdf":
            text = extract_text_from_pdf(fpath)
        else:
            text = extract_text_from_docx(fpath)
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in the document.")

    # analyze with LLM (LangChain)
    try:
        report = analyze_with_langchain(text, guidelines)
    except Exception as e:
        logger.exception("LLM analysis failed")
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")

    return JSONResponse(content={
        "filename": fname,
        "content_type": file.content_type,
        "report": report
    })
