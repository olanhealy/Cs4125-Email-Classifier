from googletrans import Translator

translator = Translator()

def translate_to_english(text):
    """Translate the given text to English."""
    # Return the original text if it's not a valid string or is empty
    if not isinstance(text, str) or text.strip() == '':
        return text  

    try:
        # Translate the text to English
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Error translating text: {text}\nError: {e}")
        # Return the original text if translation fails
        return text  
