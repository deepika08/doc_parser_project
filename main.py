from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import shutil
import uuid
import mimetypes
import PyPDF2
import docx

app = FastAPI()

# Directory to save uploads
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed MIME types
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc"
}


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF using PyPDF2"""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX using python-docx"""
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple homepage with a link to upload form."""
    return """
    <html>
        <head>
            <title>Document Parser</title>
        </head>
        <body>
            <h2>Welcome! Upload your document</h2>
            <a href='/upload-form'>Go to Upload Form</a>
        </body>
    </html>
    """


@app.get("/upload-form", response_class=HTMLResponse)
async def upload_form():
    """Return an HTML upload form."""
    return """
    <html>
        <head>
            <title>Upload Document</title>
        </head>
        <body>
            <h2>Upload a PDF or Word Document</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept=".pdf,.doc,.docx" required>
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_ext = ALLOWED_TYPES[file.content_type]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    saved_path = UPLOAD_DIR / unique_filename

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")

    extracted_text = ""
    if file_ext == ".pdf":
        extracted_text = extract_text_from_pdf(saved_path)
    elif file_ext in [".docx", ".doc"]:
        extracted_text = extract_text_from_docx(saved_path)

    return JSONResponse(content={
        "filename": unique_filename,
        "content_type": file.content_type,
        "extracted_text_preview": extracted_text[:500]
    })