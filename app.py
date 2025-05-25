import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "clf.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

# Load pre-trained model and TF-IDF vectorizer
try:
    svc_model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))
    le = pickle.load(open(ENCODER_PATH, 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() or ''  # Ensure no NoneType error
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""


# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8').strip()
    except UnicodeDecodeError:
        return file.read().decode('latin-1').strip()


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    extract_func = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt": extract_text_from_txt
    }
    return extract_func.get(file_extension, lambda x: "")(uploaded_file)


# Function to predict the category of a resume
def predict_resume_category(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()

    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            if not resume_text:
                st.error("Failed to extract text. Please upload a valid file.")
                return

            st.success("Successfully extracted the text from the uploaded resume.")
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            category = predict_resume_category(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()

