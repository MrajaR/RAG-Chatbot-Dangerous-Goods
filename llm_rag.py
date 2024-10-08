import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import pytesseract
import mysql.connector
from dotenv import load_dotenv
import torch
from pdf2image import convert_from_path
import concurrent.futures

load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LLMRAGProcessor:
    def __init__(self):
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama-3.1-70b-Versatile")
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2",
                                                            api_key=HF_TOKEN)
        self.conversation_retrieval_chain = None

        # Define the prompt template
        self.generate_source_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "you are an expert who is very proficient at elaborating and analyzing material safety datasheet based on IMDG guide. you are able to determine the goods classification, the packaging strategy and what package to be used, and lastly you are able to DECIDE whether or not it can be loaded in the ship , it is mandatory to decide with high level confidence, yes or no"
                ),
                ("human", "{source}\nbased on above text, create a detailed paragraph about its classification, the packaging strategy that should be used and what package to be used, and lastly decide whether or not it can be loaded in the ship, decide it with high level confidence, yes or no "),
            ]
        )

        self.conversation_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "you are a very helpful assistant that can talk to user regarding material safety datasheet based on international maritime dangerous goods (IMDG) in a friendly manner, you are proficient at classifying, determine the packaging of the goods, and decide with high level confidence whether or not the goods can be loaded in a ship. and also at the same time responding interactively with user using indonesian language, it is a mandatory to speak in indonesia, otherwise you'll get punished"
                ),
                ("human", "question:\n\"{input}\"\nFormat your response with HTML tags such as <h1>, <ul>, <ol>, <p> to ensure clarity and structure for web display")
            ]
        )

        self.generate_source_chain = self.generate_source_template | self.llm
        self.db_vector = None

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_uuid VARCHAR(36) NOT NULL
        );
        """
        with self.connect_to_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(create_table_query)
            connection.commit()

    def connect_to_database(self):
        return mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
            connect_timeout=10  # Increase as needed
        )

    def save_to_database(self, user_message, bot_response, user_uuid):
        query = "INSERT INTO conversation_history (user_message, bot_response, user_uuid) VALUES (%s, %s, %s)"
        with self.connect_to_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_message, bot_response, user_uuid))
            connection.commit()

    def retrieve_chat_history(self, user_uuid, limit):
        query = "SELECT user_message, bot_response FROM conversation_history WHERE user_uuid = %s ORDER BY timestamp DESC LIMIT %s"
        with self.connect_to_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_uuid, limit))
                history = cursor.fetchall()

        # Format the history into a single string
        if history:
            formatted_history = "\n".join([f"User: {user_msg}\nBot: {bot_resp}" for user_msg, bot_resp in reversed(history)])
        else:
            formatted_history = ""  # No chat history available

        return formatted_history
    
    def ocr_page(self, page):
        return pytesseract.image_to_string(page)
    
    def delete_chromadb(self):
        # Assume you have a Chroma instance `chroma_instance` and the source document `source_doc`
        # ids_to_delete = []
        try:
            self.db_vector.delete_collection()
            print("deleting the previous chroma")
        except:
            pass

        # chroma_instance.delete(ids=ids_to_delete)

    def process_uploaded_document(self, document_path, user_uuid):
        try:
            pdf_pages = convert_from_path(document_path, dpi=350)

            user_folder = os.path.join('user_txt', user_uuid)
            os.makedirs(user_folder, exist_ok=True)

            text_file_path = os.path.join(user_folder, f'{user_uuid}.txt')

            # Use concurrent futures for parallel processing
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Process pages in parallel and get OCR text for each page
                results = executor.map(self.ocr_page, pdf_pages)

                # Write the extracted text to a file
                with open(text_file_path, 'a', encoding='utf-8') as f:
                    for text in results:
                        f.write(text + "\n")

        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return False

        with open(text_file_path, 'r') as file:
            full_text = file.read()

        print(full_text)
        source_text = self.generate_source_chain.invoke({"source":full_text}).content

        # Split the full text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # Customize the chunk size as needed
            chunk_overlap=100  # Customize the overlap size as needed
        )
        chunks = text_splitter.split_text(source_text)

        # Check if the document contains relevant keywords for dangerous goods
        if any(keyword in full_text for keyword in ['safety data', 'msds', 'hazard']):
            # Create a vector store from lines of text for retrieval
            # if self.db_vector is not None:
            #     del self.db_vector.delete(ids=uuids[-1])
            #     print('menghapus chromadb lama')

            self.delete_chromadb()

            print("membuat vector db chroma baru")
            self.db_vector = Chroma.from_texts(chunks, embedding=self.embeddings)
            print("berhasil membuat chroma db baru")

            self.conversation_retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type='stuff',
                retriever=self.db_vector.as_retriever(search_type='mmr', search_kwargs={'k': 6, 'lambda_mult': 0.25}),
                return_source_documents=True,
                input_key='pertanyaan'
            )

            return True
        else:
            return False

    def process_prompt(self, prompt, user_uuid):
        formatted_prompt = self.conversation_template.format(input=prompt)
        chat_history = self.retrieve_chat_history(user_uuid, 8)

        output = self.conversation_retrieval_chain({'pertanyaan': formatted_prompt, 'chat_history': chat_history})
        answer = output['result']
        source_documents = output['source_documents']

        self.save_to_database(prompt, answer, user_uuid)

        return answer, source_documents

    def close_connection(self):
        pass  # No longer needed with the context manager approach
