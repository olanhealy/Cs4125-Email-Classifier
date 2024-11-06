from googletrans import Translator

translator = Translator()
total_count = 0
failure_count = 0

def translate_to_english(text):
    """Translate the given text to English."""
    global total_count, failure_count
    
    total_count += 1

    if not isinstance(text, str) or text.strip() == '':
        failure_count += 1
        return text

    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        failure_count += 1
        return text

def report_translation_rate():
    """Prints the success rate of translations."""
    success_rate = ((total_count - failure_count) / total_count) * 100 if total_count > 0 else 0
    print(f"Translated {success_rate:.2f}% of lines successfully.")