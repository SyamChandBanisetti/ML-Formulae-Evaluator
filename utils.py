import pytesseract
from PIL import Image
import google.generativeai as genai

# Hardcoded Gemini API key
API_KEY = "API_KEY"  # 👈 Replace with your real key

# Configure Gemini once
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Extract text from image using pytesseract
def extract_text_from_image(image_file):
    img = Image.open(image_file).convert('L')  # grayscale improves accuracy
    text = pytesseract.image_to_string(img)
    return text.strip()

# Gemini prompt for evaluating formulas
def evaluate_formulas(user_text):
    prompt = f"""
    A user submitted formulas for accuracy, precision, recall, and F1-score in machine learning.
    Extracted formulas:
    {user_text}

    Compare these to the standard formulas. For each metric, say whether it’s correct or incorrect.
    Give a final score out of 4 (1 point for each correct formula), and explain the feedback.
    """
    response = model.generate_content(prompt)
    return response.text
