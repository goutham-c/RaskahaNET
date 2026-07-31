import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemma-3-27b-it')


def analyze_situation(image_path, user_message, disaster_type, rag_context):
    """image_path can be None if there is no image (text-only chat)."""
    prompt = f"""You are a disaster response assistant for a {disaster_type}.
Relevant safety guidance:
{chr(10).join(rag_context)}

User's message: {user_message}
Based on the situation described (and the image, if provided) and the guidance above,
assess the severity and give specific, immediate safety steps. Keep it short and actionable."""

    if image_path:
        image = genai.upload_file(image_path)
        response = model.generate_content([prompt, image])
    else:
        response = model.generate_content(prompt)

    return response.text


def text_only_chat(user_message, disaster_type, rag_context):
    return analyze_situation(None, user_message, disaster_type, rag_context)
