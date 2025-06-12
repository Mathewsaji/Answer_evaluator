from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file,send_from_directory
import os
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller  # For spelling correction
from sentence_transformers import SentenceTransformer, util  # For semantic similarity
from werkzeug.utils import secure_filename
from auth import load_users, save_user, check_user
from s3_uploader import upload_to_s3
from textract_processor import analyze_document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')  # Download stopwords dataset

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for sessions

# AWS Config
BUCKET_NAME = 'answersheet-evaluation-bucket'
REGION = 'ap-south-1'

# Global Variables to store extracted data
correct_text = None
student_text = ""
correct_keywords = []  # Store correct answer keywords
student_keywords = []  # Store student answer keywords

# Folder to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Folder to store processed PDFs
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize spell checker
spell = Speller()

# Set environment variables to increase timeout for Hugging Face downloads
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes

# Load pre-trained Sentence Transformer model for semantic similarity
try:
    # Try to load the model from Hugging Face's servers
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to download model from Hugging Face: {e}")
    # Load the model from a local directory if the download fails
    model_path = "models/all-MiniLM-L6-v2"  # Path to the locally downloaded model files
    if os.path.exists(model_path):
        print("Loading model from local directory...")
        model = SentenceTransformer(model_path)
    else:
        raise RuntimeError("Model not found locally. Please download the model manually.")

# Function: Save text to PDF
def save_text_to_pdf(text, pdf_path):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.drawString(50, height - 50, "Extracted Text")
    y = height - 80
    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 14
    c.save()

# Function: Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF file."""
    text = ''
    try:
        pdf_file.seek(0)  # Reset file pointer
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + '\n'
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text.strip()

# Function: Correct spelling in text
def correct_spelling(text):
    """
    Corrects spelling errors in the text using autocorrect.
    """
    return spell(text)

# Function: Clean and extract keywords from text
def extract_keywords(text):
    """
    Extracts keywords from text by:
    1. Correcting spelling errors.
    2. Tokenizing the text.
    3. Removing stop words.
    4. Removing non-alphanumeric tokens.
    5. Converting to lowercase.
    6. Removing duplicates.
    """
    # Correct spelling errors
    corrected_text = correct_spelling(text)

    # Tokenize the text
    words = word_tokenize(corrected_text)

    # Get English stop words
    stop_words = set(stopwords.words('english'))

    # Clean tokens: remove stop words, non-alphanumeric tokens, and convert to lowercase
    keywords = [
        word.lower() for word in words
        if word.isalnum() and word.lower() not in stop_words
    ]

    # Remove duplicates
    return list(set(keywords))

# Function: Calculate semantic similarity between two texts
def calculate_semantic_similarity(text1, text2):
    """
    Calculates the semantic similarity between two texts using a pre-trained model.
    Returns a similarity score between 0 and 1.
    """
    # Encode the texts into embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    return similarity

# Home Route
@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

# Login Route
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if check_user(username, password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials.")
    return render_template("login.html")

# Signup Route
@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if len(password) < 6:
            flash("Password must be at least 6 characters long.")
        elif save_user(username, password):
            flash("Account created successfully! Please login.")
            return redirect(url_for('login'))
        else:
            flash("User already exists.")
    return render_template("signup.html")

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# About Us Route
@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("about.html")

# Extracted Text Route
@app.route('/extracted')
def extracted():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("extracted.html")

# Upload Correct Answers PDF
@app.route('/upload_correct', methods=["POST"])
def upload_correct():
    global correct_text, correct_keywords
    if 'correct_pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    correct_pdf = request.files['correct_pdf']
    if correct_pdf.filename == '':
        return jsonify({"error": "No file selected"}), 400

    correct_text = extract_text_from_pdf(correct_pdf)
    if not correct_text:
        return jsonify({"error": "Failed to extract text"}), 400

    # Extract keywords from correct answer
    correct_keywords = extract_keywords(correct_text)

    return jsonify({
        "message": "Correct answers uploaded successfully!",
        "keywords": correct_keywords
    })

# Upload Student Answer (Handwritten)
@app.route('/upload_student', methods=["POST"])
def upload_student():
    global student_text, student_keywords
    if 'student_pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    student_pdf = request.files['student_pdf']
    if student_pdf.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, student_pdf.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    student_pdf.save(file_path)

    # Upload to S3 and process with Textract
    s3_key = f"input/{student_pdf.filename}"
    upload_to_s3(BUCKET_NAME, file_path, s3_key, REGION)
    extracted_text = analyze_document(BUCKET_NAME, s3_key, REGION)

    # Correct spelling errors in the extracted text
    student_text = correct_spelling(extracted_text)
    
    # Extract keywords from student answer
    student_keywords = extract_keywords(student_text)

    # Find common keywords and missing keywords
    common_keywords = list(set(student_keywords) & set(correct_keywords))
    missing_keywords = list(set(correct_keywords) - set(student_keywords))

    # Save extracted text as PDF
    pdf_path = os.path.join(OUTPUT_FOLDER, f"{student_pdf.filename}.pdf")
    save_text_to_pdf(student_text, pdf_path)

    return jsonify({
        "message": "Student answers uploaded successfully!",
        "keywords": student_keywords,
        "common_keywords": common_keywords,
        "missing_keywords": missing_keywords
    })

# Evaluate Student Answer
@app.route('/evaluate', methods=["POST"])
def evaluate():
    global correct_text, student_text
    if correct_text is None:
        return jsonify({"error": "Upload the correct answer first"}), 400
    if not student_text:
        return jsonify({"error": "Upload the student answer first"}), 400

    # Calculate semantic similarity between correct answer and student's answer
    similarity_score = calculate_semantic_similarity(correct_text, student_text)

    # Convert similarity score to a percentage
    score = similarity_score * 100

    # Return evaluation result
    return jsonify({
        "message": f"Evaluation Complete. Score: {score:.2f}%",
        "score": score,
        "correct_text": correct_text,  # Correct answer text
        "student_text": student_text,  # Student's answer text
    })

@app.route('/get_processed_pdfs', methods=["GET"])
def get_processed_pdfs():
    if not os.path.exists(OUTPUT_FOLDER):
        return jsonify({"error": "No processed PDFs found"}), 404

    # List all PDF files in the output folder
    pdf_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".pdf")]
    return jsonify({"pdf_files": pdf_files})

# Route to download a processed PDF
@app.route('/download/<filename>')
def download(filename):
    # Ensure the file exists in the OUTPUT_FOLDER
    if not os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
        return jsonify({"error": "File not found"}), 404

    # Serve the file for download
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)



# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)