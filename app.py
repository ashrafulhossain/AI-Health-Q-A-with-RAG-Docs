from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import os
import openai
import pdfplumber
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import easyocr
import shutil

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OpenAI API Key is missing. Check your .env file.")

app = FastAPI()

# Temporary storage for uploaded PDFs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Predefined health-related questions
default_questions = [
    {"question": "Do you exercise regularly?"},
    {"question": "Do you sleep 7-8 hours a day?"},
    {"question": "Do you frequently feel stressed?"},
    {"question": "Do you have high blood pressure?"},
    {"question": "Do you consume tobacco or alcohol?"}
]

@app.get("/get_questions/")
def get_default_questions():
    return {"questions": default_questions}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    return text.strip()

# Convert PDF pages to images for OCR
def convert_pdf_to_images(pdf_path):
    image_paths = []
    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:\\Users\\STA\\Desktop\\poppler-24.08.0\\Library\\bin")
        for i, image in enumerate(images):
            image_path = f"{UPLOAD_DIR}/page_{i+1}.png"
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF to images: {str(e)}")
    return image_paths

# Extract text from images using OCR
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    extracted_text = " ".join([detection[1] for detection in result])
    return extracted_text

# Process PDFs and create FAISS index
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    image_paths = convert_pdf_to_images(pdf_path)
    for img_path in image_paths:
        text += " " + extract_text_from_image(img_path)
        os.remove(img_path)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No valid text found in PDF.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Store FAISS Index in memory
vector_store = None

@app.post("/upload_pdf/")
def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    vector_store = process_pdf(file_path)
    os.remove(file_path)  # Clean up after processing
    return {"message": "PDF processed successfully! You can now ask questions."}

@app.post("/ask/")
def ask_question(question: str = Form(...)):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No PDF processed yet. Please upload a PDF first.")
    
    docs = vector_store.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs])
    
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers."},
            {"role": "user", "content": f"User asked: {question}\n\nScientific context: {context}"}
        ],
        max_tokens=500,
        temperature=0.1
    )
    
    return {"answer": response.choices[0].message.content.strip()}
