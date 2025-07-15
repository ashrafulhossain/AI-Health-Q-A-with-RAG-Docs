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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ OpenAI API Key is missing. Check your .env file.")
    exit()

# Predefined health questions
def get_default_questions_and_answers():
    return [
        {"question": "Do you exercise regularly?", "answer": ""},
        {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
        {"question": "Do you frequently feel stressed?", "answer": ""},
        {"question": "Do you have high blood pressure?", "answer": ""},
        {"question": "Do you consume tobacco or alcohol?", "answer": ""}
    ]

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
    return text.strip()

# Convert PDF pages to images for OCR
def convert_pdf_to_images(pdf_path):
    image_paths = []
    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")
        for i, image in enumerate(images):
            image_path = f"page_{i+1}.png"
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
    except Exception as e:
        print(f"❌ Error converting {pdf_path} to images: {e}")
    return image_paths

# Extract text from images using OCR
def extract_text_from_image(image_path):
    print(f"🔍 Running OCR on {image_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    extracted_text = " ".join([detection[1] for detection in result])
    print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")
    return extracted_text

# Process PDFs with OCR support
def process_multiple_pdfs(pdf_files):
    all_text = ""

    for pdf_file in pdf_files:
        print(f"📄 Processing PDF: {pdf_file}")
        
        text = extract_text_from_pdf(pdf_file)
        print(f"✅ Extracted text length from {pdf_file}: {len(text)} characters")
        all_text += text + " "

        # Extract images and run OCR
        print("📸 Extracting images and running OCR...")
        image_paths = convert_pdf_to_images(pdf_file)
        for img_path in image_paths:
            ocr_text = extract_text_from_image(img_path)
            all_text += ocr_text + " "
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"⚠️ Error deleting {img_path}: {e}")

    if not all_text.strip():
        print("❌ No valid text found in PDFs.")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(all_text)
    documents = [Document(page_content=text) for text in texts]

    # Create FAISS vector store
    print("✅ Generating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Collect health history from the user
def collect_health_history():
    print("📝 Please answer the following questions to help us provide personalized advice:")
    health_history = {
        "age": input("Your age: "),
        "gender": input("Your gender (Male/Female/Other): "),
        "medical_conditions": input("Any existing medical conditions (e.g., diabetes, asthma): "),
        "medications": input("Are you currently taking any medications? If yes, please list them: "),
        "allergies": input("Do you have any allergies? If yes, please specify: ")
    }
    return health_history

# Get AI response while maintaining chat history
def get_answer_from_llm(user_input, vector_store, chat_history, health_history):
    try:
        docs = vector_store.similarity_search(user_input)
        context = " ".join([doc.page_content for doc in docs])

        # Add health history to the context
        health_context = f"User's health history: {health_history}\n\n"
        full_context = health_context + context

        chat_history.append({"role": "user", "content": user_input})

        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Always mention the original source, expert name, or research study if available."},
                *chat_history,
                {"role": "user", "content": f"User asked: {user_input}\n\nHealth context: {health_context}\n\nScientific context: {full_context}"}
            ],
            max_tokens=500  # Increased max_tokens for complete responses
        )

        ai_response = response.choices[0].message.content.strip()
        
        # Check if the response is cut off
        if "..." in ai_response or "..." in ai_response[-10:]:  # Simple check for incomplete responses
            ai_response += "\n\n[Note: Response may be incomplete due to token limits.]"

        chat_history.append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Error generating response."

# Main function to handle conversation
def chat_with_bot(pdf_files):
    print("🚀 Starting AI Chatbot...")
    vector_store = process_multiple_pdfs(pdf_files)
    if not vector_store:
        return

    chat_history = []
    health_history = collect_health_history()  # Collect health history
    questions = get_default_questions_and_answers()

    # Collect user answers
    user_responses = []
    for qa in questions:
        print(f"\n❓ {qa['question']}")
        answer = input("Your answer (yes/no/details): ")
        user_responses.append(f"{qa['question']} {answer}")
        qa['answer'] = answer

    # Initial AI response
    print("\n🎯 Generating AI response...")
    combined_answers = " | ".join(user_responses)
    ai_response = get_answer_from_llm(combined_answers, vector_store, chat_history, health_history)
    print("\n✅ AI Recommendations:")
    print("----------------------------")
    print(ai_response)

    # Continue chatting
    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Chatbot session ended.")
            break

        ai_response = get_answer_from_llm(user_input, vector_store, chat_history, health_history)
        print(f"\n🤖 AI: {ai_response}")

# Example usage
pdf_files = [r"C:\Users\STA\Desktop\Chuxy\ABPM vs office in HTN_NEJM.pdf"]
chat_with_bot(pdf_files)