from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import logging
import os
from llm_rag import LLMRAGProcessor
import uuid

app = Flask(__name__)
app.secret_key = 'SPIL'  # Necessary for session management
cors = CORS(app, resources={r'/*': {'origins': '*'}})
app.logger.setLevel(logging.ERROR)

processor = LLMRAGProcessor()

# Create a directory for user PDFs if it doesn't exist
if not os.path.exists('user_pdf'):
    os.makedirs('user_pdf')

@app.before_request
def initialize_db():
    # Ensure the table exists before handling any requests
    processor.create_table_if_not_exists()

@app.before_request
def assign_user_id():
    # Assign a unique ID to each user if not already assigned
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        print(session['user_id'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-message', methods=['POST'])
def process_prompt_route():
    user_prompt = request.json.get('userMessage')
    user_id = session['user_id']

    if not user_prompt:
        return jsonify({'error': 'Tolong Masukkan Prompt.'}), 400

    print('user message:', user_prompt)

    # need to be change if wanting to use msds only
    # Pass the user_id to the processor to handle user-specific data
    response, msds_source_documents = processor.process_prompt(user_prompt, user_id)

    print('source document:', msds_source_documents)

    return jsonify({'botResponse': response}), 200

@app.route('/process-document', methods=['POST'])
def process_document_route():
    user_id = session['user_id']

    if 'file' not in request.files:
        return jsonify({'botResponse': 'Tolong upload file PDF'}), 400

    file = request.files['file']

    # Store file in a user-specific directory
    user_dir_pdf = os.path.join("user_pdf", user_id)
    os.makedirs(user_dir_pdf, exist_ok=True)

    # Remove any existing files in the user's directory
    for existing_file in os.listdir(user_dir_pdf):
        existing_file_path = os.path.join(user_dir_pdf, existing_file)
        os.remove(existing_file_path)

    user_dir_txt = os.path.join("user_txt", user_id)

    # Remove any existing txt files in the each user dir
    if os.path.exists(user_dir_txt):
        for existing_txt in os.listdir(user_dir_txt):
            existing_txt_path = os.path.join(user_dir_txt, existing_txt)
            os.remove(existing_txt_path)

    # Save the new file
    msds_doc_path = os.path.join(user_dir_pdf, file.filename)
    file.save(msds_doc_path)

    # Process the uploaded document for the specific user
    is_process_ok = processor.process_uploaded_document(msds_doc_path, user_id)

    if is_process_ok:
        return jsonify({'botResponse': 'Terima kasih telah mengirimkan PDF, saya telah menganalisis dokumen MSDS nya, silahkan bertanya apapun terkait dengan dokumen tersebut'}), 200
    else:
        os.remove(msds_doc_path)
        return jsonify({'botResponse': 'Maaf, saya tidak bisa memproses dokumen yang bukan merupakan dokumen MSDS'}), 400
    
@app.teardown_appcontext
def close_db_connection(Exception=None):
    processor.close_connection()
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
