import os
import logging
from groq import Groq
from classes.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
os.environ['GROQ_API_KEY'] = settings.get_setting('groq_api_key')
model_name = settings.get_setting('groq_model')

groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

def improve_text(chunk):
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""Please edit this text and Follow these steps to edit the text:
            Carefully read through the text and identify any grammatical, spelling, or punctuation errors. Correct these errors while maintaining the original word choice as much as possible.
            Improve the readability of the text by:
            a. Breaking up overly long sentences into shorter, clearer ones.
            b. Rearranging words or phrases for better flow and clarity.
            c. Replacing unclear or awkward phrasing with more natural alternatives.
            d. Ensuring consistent tense usage throughout the text.
            While making these improvements, be careful to preserve the original meaning and style of the text. Do not add new information or change the author's intent. Do not leave any comments.
            If you encounter any specialized terminology or proper nouns, assume they are correct unless there is an obvious spelling error.
            After editing, review your changes to ensure they enhance the text without altering its core message or tone.
            Provide your edited version of the text within <edited_text> tags. Do not include any explanations, comments, or lists of changes made.
            Remember, your goal is to improve the text while keeping it as close to the original as possible. Make only necessary changes to correct errors and enhance readability.
            IMPORTANT: Always respond in the language of the original text. Do not translate or switch to any other language under any circumstances.
            Here is the original text you will be working with: {chunk}""",
                }
            ],
            model=model_name,
        )
        
        improved_text = response.choices[0].message.content
        return improved_text
    
    except Exception as e:
        logger.error(f"Error improving text: {e}")
        return None