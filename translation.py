from langdetect import detect
from googletrans import Translator

def translate_medicine_text(text):
    """Translate non-English medicine text to English"""
    try:
        if len(text.strip()) < 3:  # Skip short texts
            return text
            
        lang = detect(text)
        if lang != 'en':
            translator = Translator()
            return translator.translate(text, dest='en').text
        return text
    except:
        return text  # Fallback to original text
