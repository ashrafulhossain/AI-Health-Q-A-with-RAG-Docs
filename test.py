# import openai
# import faiss
# import numpy as np
# import pdfplumber
# import fitz  # PyMuPDF
# import easyocr
# from pdf2image import convert_from_path
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# # Predefined health-related questions
# def get_default_questions_and_answers():
#     default_data = [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]
#     return default_data


# # Step 1: Extract text from PDF using pdfplumber
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text


# # Step 2: Extract images from PDF using PyMuPDF
# def extract_images_from_pdf(pdf_path):
#     images = []
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         pix = page.get_pixmap()
#         img_path = f"page_{page_num+1}.png"
#         pix.save(img_path)
#         images.append(img_path)
#     return images


# # Step 3: Convert PDF pages to images using pdf2image
# def convert_pdf_to_images(pdf_path):
#     images = convert_from_path(pdf_path)
#     image_paths = []
#     for i, image in enumerate(images):
#         image_path = f"page_{i+1}.png"
#         image.save(image_path, 'PNG')
#         image_paths.append(image_path)
#     return image_paths


# # Step 4: Extract text from images using OCR (Using EasyOCR)
# def extract_text_from_image(image_path):
#     reader = easyocr.Reader(['en'])  # 'en' for English language OCR
#     result = reader.readtext(image_path)
#     text = " ".join([detection[1] for detection in result])  # Join the extracted text
#     return text


# # Step 5: Using LangChain OpenAI Embeddings to generate embeddings
# def get_embeddings(text):
#     openai_api_key = 'your_openai_api_key'  # Replace with your OpenAI API key
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     return embeddings.embed_query(text)


# # Step 6: Create FAISS Index using LangChain FAISS
# def create_faiss_index(embeddings):
#     index = FAISS.from_documents([Document(page_content=embeddings)])
#     faiss.normalize_L2(np.array(embeddings, dtype=np.float32))  # Normalizing embeddings for better search
#     index.add(np.array(embeddings, dtype=np.float32))  # Adding embeddings to the index
#     return index


# # Step 7: Search the FAISS Index for similarity
# def search_faiss_index(query, index):
#     query_vector = get_embeddings(query)
#     D, I = index.search(query_vector.astype(np.float32), k=1)
#     return D, I


# # Step 8: Get answer from GPT-4
# def get_answer_from_llm(query, index):
#     D, I = search_faiss_index(query, index)
#     response = openai.ChatCompletion.create(
#         model="gpt-4",  # Use GPT-4 model
#         messages=[{"role": "user", "content": query}]
#     )
#     answer = response['choices'][0]['message']['content']
#     return answer


# # Function to ask questions and get user responses one by one
# def ask_questions_and_generate_answer(pdf_files):
#     # Process all PDFs and create a FAISS index
#     index = process_multiple_pdfs(pdf_files)
    
#     # Collecting user responses
#     user_responses = []
#     default_data = get_default_questions_and_answers()
    
#     # Ask each question and store the response
#     for idx, qa in enumerate(default_data):
#         print(f"Question {idx + 1}: {qa['question']}")
#         user_answer = input("Your answer: ")  # User input
#         user_responses.append(user_answer)  # Collect the answer
        
#         # Save the user's answer to the current question
#         qa['answer'] = user_answer
        
#     # After all questions are answered, generate the AI response
#     print("Based on your answers, the AI model is giving advice:")
    
#     # Combine all responses into one text
#     combined_answers = " ".join(user_responses)
    
#     # Get the AI answer using the combined responses
#     answer = get_answer_from_llm(combined_answers, index)
#     print(answer)


# # Process all PDFs, extract text and images, and store embeddings in FAISS
# def process_multiple_pdfs(pdf_files):
#     all_text = ""
#     all_images = []
#     for pdf_file in pdf_files:
#         print(f"Processing PDF: {pdf_file}")  # Log the PDF being processed
#         text = extract_text_from_pdf(pdf_file)
#         images = extract_images_from_pdf(pdf_file)
#         all_text += text
#         all_images.extend(images)
    
#     # Extract text from images using OCR
#     all_image_text = [extract_text_from_image(image) for image in all_images]
    
#     # Merge all text together
#     merged_data = all_text + " " + " ".join(all_image_text)

#     # Step 1: Split the text into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents([Document(page_content=merged_data)])

#     # Step 2: Create embeddings using LangChain OpenAI
#     embeddings = get_embeddings(merged_data)
    
#     # Step 3: Create Faiss Index
#     index = create_faiss_index(embeddings)
    
#     return index


# # List of PDF files to process
# pdf_files = [
#     r"C:\Users\STA\Desktop\Chuxy\ABPM vs office in HTN_NEJM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Cost savings of ABPM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Lee2022_clinical decisions remote BPM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\NIGHTIME ABPM nightime dippers.risers.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\OptiBP app.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\photophytoelectric signal for BP.pdf"
# ]

# # Run the program
# ask_questions_and_generate_answer(pdf_files)

















# import openai
# import faiss
# import numpy as np
# import pdfplumber
# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")  # Using OPENAI_API_KEY from .env file

# # Ensure OpenAI API Key is loaded
# if not openai_api_key:
#     print("❌ OpenAI API Key is missing. Check your .env file.")
#     exit()

# # Predefined health-related questions
# def get_default_questions_and_answers():
#     return [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]

# # Step 1: Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#     except Exception as e:
#         print(f"❌ Error extracting text from {pdf_path}: {e}")
#     return text.strip()

# # Step 2: Extract images from PDF
# def extract_images_from_pdf(pdf_path):
#     images = []
#     try:
#         doc = fitz.open(pdf_path)
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             pix = page.get_pixmap()
#             img_path = f"page_{page_num+1}.png"
#             pix.save(img_path)
#             images.append(img_path)
#     except Exception as e:
#         print(f"❌ Error extracting images from {pdf_path}: {e}")
#     return images

# # Step 3: Convert PDF pages to images (Still enabled, but OCR skipped)
# def convert_pdf_to_images(pdf_path):
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path)
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"❌ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Step 4: OCR is **SKIPPED**
# def extract_text_from_image(image_path):
#     print(f"🔍 Skipping OCR for debugging...")  # OCR Disabled
#     return ""

# # Step 5: Generate OpenAI Embeddings
# def get_embeddings(text):
#     try:
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         return embeddings.embed_query(text)
#     except Exception as e:
#         print(f"❌ Error generating embeddings: {e}")
#         return None

# # Step 6: Create FAISS Index
# def create_faiss_index(embeddings):
#     try:
#         print("✅ Creating FAISS index...")
#         if not embeddings:
#             print("❌ No embeddings found, FAISS index creation failed.")
#             return None

#         print(f"✅ Embeddings received with length: {len(embeddings)}")  # Debugging
#         index = faiss.IndexFlatL2(len(embeddings))
#         np_embeddings = np.array([embeddings], dtype=np.float32)

#         print("✅ Normalizing embeddings...")
#         faiss.normalize_L2(np_embeddings)

#         print("✅ Adding embeddings to FAISS index...")
#         index.add(np_embeddings)

#         print("✅ FAISS index created successfully!")
#         return index
#     except Exception as e:
#         print(f"❌ Error creating FAISS index: {e}")
#         return None

# # Step 7: Search FAISS Index
# def search_faiss_index(query, index):
#     try:
#         print(f"🔍 Searching FAISS index for query: {query}")
#         query_vector = get_embeddings(query)
#         if query_vector is None:
#             return None, None
#         D, I = index.search(np.array([query_vector], dtype=np.float32), k=1)
#         print(f"✅ Search results: {D}, {I}")
#         return D, I
#     except Exception as e:
#         print(f"❌ Error searching FAISS index: {e}")
#         return None, None

# # Step 8: Get answer from GPT-4 (Updated for new API)
# def get_answer_from_llm(query, index):
#     try:
#         D, I = search_faiss_index(query, index)
#         if D is None or I is None:
#             print("❌ No results from FAISS index. Returning default response.")
#             return "I'm unable to find relevant information. Please try again."

#         # Updated OpenAI API call
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo" depending on your access
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=300  # Adjust max_tokens as needed
#         )
#         answer = response.choices[0].message.content.strip()
#         return answer

#     except Exception as e:
#         print(f"❌ Error getting answer from GPT-4: {e}")
#         return "Error generating response. Please try again."

# # Step 9: Process PDFs and create FAISS index
# def process_multiple_pdfs(pdf_files):
#     all_text = ""

#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
#         text = extract_text_from_pdf(pdf_file)
#         print(f"✅ Extracted text length from {pdf_file}: {len(text)} characters")  # Debugging
#         all_text += text + " "

#     print("✅ Merging text and skipping OCR...")
#     merged_data = all_text  # OCR skipped

#     if not merged_data.strip():
#         print("❌ No valid text found in PDFs.")
#         return None

#     print("✅ Generating embeddings...")
#     embeddings = get_embeddings(merged_data)

#     if embeddings is None:
#         print("❌ Error generating embeddings.")
#         return None

#     print("✅ Creating FAISS index...")
#     index = create_faiss_index(embeddings)
#     return index

# # Step 10: Ask Questions and Generate Answer
# def ask_questions_and_generate_answer(pdf_files):
#     print("🚀 Starting process...")
#     index = process_multiple_pdfs(pdf_files)
#     if index is None:
#         print("❌ FAISS index creation failed.")
#         return

#     user_responses = []
#     questions = get_default_questions_and_answers()

#     for idx, qa in enumerate(questions):
#         print(f"❓ Question {idx + 1}: {qa['question']}")
#         user_answer = input("Your answer: ")
#         user_responses.append(user_answer)
#         qa['answer'] = user_answer

#     print("✅ Collecting all responses. Querying AI...")
#     combined_answers = " ".join(user_responses)
#     answer = get_answer_from_llm(combined_answers, index)

#     print("\n🎯 AI Response:")
#     print(answer)

# # Example Usage (Your PDFs)
# pdf_files = [
#     r"C:\Users\STA\Desktop\Chuxy\ABPM vs office in HTN_NEJM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Cost savings of ABPM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\Lee2022_clinical decisions remote BPM.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\NIGHTIME ABPM nightime dippers.risers.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\OptiBP app.pdf", 
#     r"C:\Users\STA\Desktop\Chuxy\photophytoelectric signal for BP.pdf"
# ]

# ask_questions_and_generate_answer(pdf_files)




