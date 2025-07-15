# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OpenMP সংঘর্ষ সমাধান

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
# import easyocr  # Importing easyocr for OCR functionality

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

# # Step 3: Convert PDF pages to images (Still enabled, but OCR will be used)
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

# # Step 4: Extract text from images using OCR
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on {image_path}...")  # Enabling OCR
#     reader = easyocr.Reader(['en'], gpu=False)  # Ensure OCR works on CPU
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")  # Debugging
#     return extracted_text

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

#         # নতুন OpenAI API কল
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",  # GPT-4 বা GPT-3.5 ব্যবহার করুন
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=150  # টোকেন সংখ্যা ঠিক করুন
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




# import os
# import openai
# import pdfplumber
# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr  # Importing easyocr for OCR functionality

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

# # Step 3: Convert PDF pages to images (Still enabled, but OCR will be used)
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

# # Step 4: Extract text from images using OCR
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on {image_path}...")  # Enabling OCR
#     reader = easyocr.Reader(['en'], gpu=False)  # Ensure OCR works on CPU
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")  # Debugging
#     return extracted_text

# # Step 5: Process PDFs and create FAISS index
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

#     # Split text into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,  # Maximum size of each chunk
#         chunk_overlap=200  # Overlap between chunks
#     )
#     texts = text_splitter.split_text(merged_data)

#     # Convert text chunks into Document objects
#     documents = [Document(page_content=text) for text in texts]

#     print("✅ Generating embeddings...")
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#     print("✅ Creating FAISS index...")
#     vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store

# # Step 6: Get answer from GPT-4 (Updated for new API)
# def get_answer_from_llm(query, vector_store):
#     try:
#         # Search the FAISS vector database
#         docs = vector_store.similarity_search(query)
#         context = " ".join([doc.page_content for doc in docs])

#         # New OpenAI API call
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",  # Use "gpt-4" or "gpt-3.5-turbo"
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"{query}\n\nContext: {context}"}
#             ],
#             max_tokens=150  # Adjust max_tokens as needed
#         )
#         answer = response.choices[0].message.content.strip()
#         return answer

#     except Exception as e:
#         print(f"❌ Error getting answer from GPT-4: {e}")
#         return "Error generating response. Please try again."

# # Step 7: Ask Questions and Generate Answer
# def ask_questions_and_generate_answer(pdf_files):
#     print("🚀 Starting process...")
#     vector_store = process_multiple_pdfs(pdf_files)
#     if vector_store is None:
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
#     answer = get_answer_from_llm(combined_answers, vector_store)

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





# import os
# import openai
# import pdfplumber
# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr  # For OCR

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     print("❌ OpenAI API Key is missing. Check your .env file.")
#     exit()

# # Predefined health questions
# def get_default_questions_and_answers():
#     return [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#     except Exception as e:
#         print(f"❌ Error extracting text from {pdf_path}: {e}")
#     return text.strip()

# # Convert PDF pages to images (for OCR)
# def convert_pdf_to_images(pdf_path):
#     image_paths = []
#     try:
#         # Set poppler path if not in system PATH
#         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")  # MODIFY THIS PATH
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"❌ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Extract text from images using OCR
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")
#     return extracted_text

# # Process PDFs with OCR support
# def process_multiple_pdfs(pdf_files):
#     all_text = ""

#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
        
#         # Extract regular text
#         text = extract_text_from_pdf(pdf_file)
#         print(f"✅ Extracted text length from {pdf_file}: {len(text)} characters")
#         all_text += text + " "

#         # Extract images and run OCR (NEW CODE ADDED)
#         print("📸 Extracting images and running OCR...")
#         image_paths = convert_pdf_to_images(pdf_file)
#         for img_path in image_paths:
#             ocr_text = extract_text_from_image(img_path)
#             all_text += ocr_text + " "
#             try:
#                 os.remove(img_path)  # Clean up temporary images
#             except Exception as e:
#                 print(f"⚠️ Error deleting {img_path}: {e}")

#     if not all_text.strip():
#         print("❌ No valid text found in PDFs.")
#         return None

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     texts = text_splitter.split_text(all_text)
#     documents = [Document(page_content=text) for text in texts]

#     # Create FAISS vector store
#     print("✅ Generating embeddings and FAISS index...")
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store

# # Get answer from GPT-4
# def get_answer_from_llm(query, vector_store):
#     try:
#         docs = vector_store.similarity_search(query)
#         context = " ".join([doc.page_content for doc in docs])

#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide concise recommendations based on the context."},
#                 {"role": "user", "content": f"User responses: {query}\n\nMedical context: {context}"}
#             ],
#             max_tokens=200
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return "Error generating response."

# # Main function to ask questions
# def ask_questions_and_generate_answer(pdf_files):
#     print("🚀 Starting analysis...")
#     vector_store = process_multiple_pdfs(pdf_files)
#     if not vector_store:
#         return

#     user_responses = []
#     questions = get_default_questions_and_answers()

#     for qa in questions:
#         print(f"\n❓ {qa['question']}")
#         answer = input("Your answer (yes/no/details): ")
#         user_responses.append(f"{qa['question']} {answer}")
#         qa['answer'] = answer

#     print("\n🎯 Generating AI response...")
#     combined_answers = " | ".join(user_responses)
#     ai_response = get_answer_from_llm(combined_answers, vector_store)
#     print("\n✅ AI Recommendations:")
#     print("----------------------------")
#     print(ai_response)

# # Example usage
# pdf_files = [
#     r"C:\Users\STA\Desktop\Chuxy\ABPM vs office in HTN_NEJM.pdf",
#     # Add more PDF paths here
# ]

# ask_questions_and_generate_answer(pdf_files)







# import os
# import openai
# import pdfplumber
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key:
#     print("❌ OpenAI API Key is missing. Check your .env file.")
#     exit()

# # Predefined health questions
# def get_default_questions_and_answers():
#     return [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#     except Exception as e:
#         print(f"❌ Error extracting text from {pdf_path}: {e}")
#     return text.strip()

# # Convert PDF pages to images for OCR
# def convert_pdf_to_images(pdf_path):
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"❌ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Extract text from images using OCR
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")
#     return extracted_text

# # Process PDFs with OCR support
# def process_multiple_pdfs(pdf_files):
#     all_text = ""

#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
        
#         text = extract_text_from_pdf(pdf_file)
#         print(f"✅ Extracted text length from {pdf_file}: {len(text)} characters")
#         all_text += text + " "

#         # Extract images and run OCR
#         print("📸 Extracting images and running OCR...")
#         image_paths = convert_pdf_to_images(pdf_file)
#         for img_path in image_paths:
#             ocr_text = extract_text_from_image(img_path)
#             all_text += ocr_text + " "
#             try:
#                 os.remove(img_path)
#             except Exception as e:
#                 print(f"⚠️ Error deleting {img_path}: {e}")

#     if not all_text.strip():
#         print("❌ No valid text found in PDFs.")
#         return None

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
#     documents = [Document(page_content=text) for text in texts]

#     # Create FAISS vector store
#     print("✅ Generating embeddings and FAISS index...")
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store

# # Get AI response while maintaining chat history
# def get_answer_from_llm(user_input, vector_store, chat_history):
#     try:
#         docs = vector_store.similarity_search(user_input)
#         context = " ".join([doc.page_content for doc in docs])

#         chat_history.append({"role": "user", "content": user_input})

#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
#                 *chat_history,
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.1
#         )

#         ai_response = response.choices[0].message.content.strip()
#         chat_history.append({"role": "assistant", "content": ai_response})
#         return ai_response
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return "Error generating response."

# # Main function to handle conversation
# def chat_with_bot(pdf_files):
#     print("🚀 Starting AI Chatbot...")
#     vector_store = process_multiple_pdfs(pdf_files)
#     if not vector_store:
#         return

#     chat_history = []
#     questions = get_default_questions_and_answers()

#     # Collect user answers
#     user_responses = []
#     for qa in questions:
#         print(f"\n❓ {qa['question']}")
#         answer = input("Your answer (yes/no/details): ")
#         user_responses.append(f"{qa['question']} {answer}")
#         qa['answer'] = answer

#     # Initial AI response
#     print("\n🎯 Generating AI response...")
#     combined_answers = " | ".join(user_responses)
#     ai_response = get_answer_from_llm(combined_answers, vector_store, chat_history)
#     print("\n✅ AI Recommendations:")
#     print("----------------------------")
#     print(ai_response)

#     # Continue chatting
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break

#         ai_response = get_answer_from_llm(user_input, vector_store, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Example usage
# pdf_files = [
#     r"C:\Users\STA\Desktop\Chuxy\Cost savings of ABPM.pdf",
#     r"C:\Users\STA\Desktop\Chuxy\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf",
#     r"C:\Users\STA\Desktop\Chuxy\Lee2022_clinical decisions remote BPM.pdf",
#     r"C:\Users\STA\Desktop\Chuxy\NIGHTIME ABPM nightime dippers.risers.pdf",
#     r"C:\Users\STA\Desktop\Chuxy\OptiBP app.pdf",
#     r"C:\Users\STA\Desktop\Chuxy\photophytoelectric signal for BP.pdf"
# ]

# chat_with_bot(pdf_files)


# import os
# import openai
# import pdfplumber
# from pdf2image import convert_from_path
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import easyocr

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key:
#     print("❌ OpenAI API Key is missing. Check your .env file.")
#     exit()

# # Define maximum chat history length
# MAX_HISTORY_LENGTH = 10  # You can adjust this value as needed

# # Predefined health questions
# def get_default_questions_and_answers():
#     return [
#         {"question": "Do you exercise regularly?", "answer": ""},
#         {"question": "Do you sleep 7-8 hours a day?", "answer": ""},
#         {"question": "Do you frequently feel stressed?", "answer": ""},
#         {"question": "Do you have high blood pressure?", "answer": ""},
#         {"question": "Do you consume tobacco or alcohol?", "answer": ""}
#     ]

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#     except Exception as e:
#         print(f"❌ Error extracting text from {pdf_path}: {e}")
#     return text.strip()

# # Convert PDF pages to images for OCR
# def convert_pdf_to_images(pdf_path):
#     image_paths = []
#     try:
#         images = convert_from_path(pdf_path, poppler_path=r"C:\Users\STA\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")
#         for i, image in enumerate(images):
#             image_path = f"page_{i+1}.png"
#             image.save(image_path, 'PNG')
#             image_paths.append(image_path)
#     except Exception as e:
#         print(f"❌ Error converting {pdf_path} to images: {e}")
#     return image_paths

# # Extract text from images using OCR
# def extract_text_from_image(image_path):
#     print(f"🔍 Running OCR on {image_path}...")
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path)
#     extracted_text = " ".join([detection[1] for detection in result])
#     print(f"✅ OCR extracted {len(extracted_text)} characters from {image_path}")
#     return extracted_text

# # Process PDFs with OCR support
# def process_multiple_pdfs(pdf_files):
#     all_text = ""

#     for pdf_file in pdf_files:
#         print(f"📄 Processing PDF: {pdf_file}")
        
#         text = extract_text_from_pdf(pdf_file)
#         print(f"✅ Extracted text length from {pdf_file}: {len(text)} characters")
#         all_text += text + " "

#         # Extract images and run OCR
#         print("📸 Extracting images and running OCR...")
#         image_paths = convert_pdf_to_images(pdf_file)
#         for img_path in image_paths:
#             ocr_text = extract_text_from_image(img_path)
#             all_text += ocr_text + " "
#             try:
#                 os.remove(img_path)
#             except Exception as e:
#                 print(f"⚠️ Error deleting {img_path}: {e}")

#     if not all_text.strip():
#         print("❌ No valid text found in PDFs.")
#         return None

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(all_text)
#     documents = [Document(page_content=text) for text in texts]

#     # Create FAISS vector store
#     print("✅ Generating embeddings and FAISS index...")
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store

# # Get AI response while maintaining chat history
# def get_answer_from_llm(user_input, vector_store, chat_history):
#     try:
#         # Add user's question to chat history
#         chat_history.append({"role": "user", "content": user_input})

#         # Trim chat history if it exceeds the maximum length
#         if len(chat_history) > MAX_HISTORY_LENGTH:
#             chat_history = chat_history[-MAX_HISTORY_LENGTH:]  # Keep only the last MAX_HISTORY_LENGTH messages

#         # Get relevant context from FAISS vector store
#         docs = vector_store.similarity_search(user_input)
#         context = " ".join([doc.page_content for doc in docs])

#         # Send chat history and context to GPT-4
#         client = openai.OpenAI(api_key=openai_api_key)
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
#                 *chat_history,  # Send trimmed chat history
#                 {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
#             ],
#             max_tokens=500,
#             temperature=0.1
#         )

#         # Get AI's response
#         ai_response = response.choices[0].message.content.strip()
        
#         # Add AI's response to chat history
#         chat_history.append({"role": "assistant", "content": ai_response})
        
#         return ai_response
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return "Error generating response."

# # Main function to handle conversation
# def chat_with_bot(pdf_files):
#     print("🚀 Starting AI Chatbot...")
#     vector_store = process_multiple_pdfs(pdf_files)
#     if not vector_store:
#         return

#     chat_history = []
#     questions = get_default_questions_and_answers()

#     # Collect user answers
#     user_responses = []
#     for qa in questions:
#         print(f"\n❓ {qa['question']}")
#         answer = input("Your answer (yes/no/details): ")
#         user_responses.append(f"{qa['question']} {answer}")
#         qa['answer'] = answer

#     # Initial AI response
#     print("\n🎯 Generating AI response...")
#     combined_answers = " | ".join(user_responses)
#     ai_response = get_answer_from_llm(combined_answers, vector_store, chat_history)
#     print("\n✅ AI Recommendations:")
#     print("----------------------------")
#     print(ai_response)

#     # Continue chatting
#     while True:
#         user_input = input("\n💬 You: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("👋 Chatbot session ended.")
#             break

#         ai_response = get_answer_from_llm(user_input, vector_store, chat_history)
#         print(f"\n🤖 AI: {ai_response}")

# # Example usage
# pdf_files = [  
#     r"C:\Users\STA\Desktop\Chuxy\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf",
# ]

# chat_with_bot(pdf_files)






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

# Define maximum chat history length
MAX_HISTORY_LENGTH = 10  # Adjust as needed

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
        
        # Extract text directly from PDF
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
                os.remove(img_path)  # Clean up image files
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

# Get AI response while maintaining chat history
def get_answer_from_llm(user_input, vector_store, chat_history):
    try:
        # Add user's question to chat history
        chat_history.append({"role": "user", "content": user_input})

        # Trim chat history if it exceeds the maximum length
        if len(chat_history) > MAX_HISTORY_LENGTH:
            chat_history = chat_history[-MAX_HISTORY_LENGTH:]  # Keep only the last MAX_HISTORY_LENGTH messages

        # Get relevant context from FAISS vector store
        docs = vector_store.similarity_search(user_input)
        context = " ".join([doc.page_content for doc in docs])

        # Send chat history and context to GPT-4
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical advisor. Provide scientific and credible answers. Mention the original source or expert name if available."},
                *chat_history,  # Send trimmed chat history
                {"role": "user", "content": f"User asked: {user_input}\n\nScientific context: {context}"}
            ],
            max_tokens=500,
            temperature=0.1
        )

        # Get AI's response
        ai_response = response.choices[0].message.content.strip()
        
        # Add AI's response to chat history
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
    ai_response = get_answer_from_llm(combined_answers, vector_store, chat_history)
    print("\n✅ AI Recommendations:")
    print("----------------------------")
    print(ai_response)

    # Continue chatting
    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Chatbot session ended.")
            break

        ai_response = get_answer_from_llm(user_input, vector_store, chat_history)
        print(f"\n🤖 AI: {ai_response}")

# Example usage
pdf_files = [
    r"C:\Users\STA\Desktop\Chuxy\ABPM vs office in HTN_NEJM.pdf",
    r"C:\Users\STA\Desktop\Chuxy\Clinical Cardiology - October 1992 - Pickering - Ambulatory blood pressure monitoring An historical perspective.pdf",
    r"C:\Users\STA\Desktop\Chuxy\Cost savings of ABPM.pdf",
    r"C:\Users\STA\Desktop\Chuxy\jamacardiology_blood_2022_oi_220067_1672335582.056.pdf",
    r"C:\Users\STA\Desktop\Chuxy\Lee2022_clinical decisions remote BPM.pdf",
    r"C:\Users\STA\Desktop\Chuxy\NIGHTIME ABPM nightime dippers.risers.pdf",
    r"C:\Users\STA\Desktop\Chuxy\OptiBP app.pdf",
    r"C:\Users\STA\Desktop\Chuxy\photophytoelectric signal for BP.pdf"
]

chat_with_bot(pdf_files)