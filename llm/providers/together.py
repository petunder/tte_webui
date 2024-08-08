import logging
from together import Together
from classes.settings import Settings
import os

settings = Settings()
model = settings.get_setting('togetherai_model')

logger = logging.getLogger(__name__)
os.environ['TOGETHER_API_KEY'] = settings.get_setting('together_api_key')

def process_chunk(chunk):
    logger.debug(f"Processing chunk with TogetherAI. Chunk length: {len(chunk)}")

    messages = [
        {"role": "system", "content": """You are an experienced editor tasked with improving a given text. Your goal is to correct errors and enhance readability while staying close to the original text and preserving its initial meaning.
            Follow these steps to edit the text:
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
            Here is the original text you will be working with:"""},
        {"role": "user", "content": chunk}
    ]

    try:
        together_ai = Together(api_key=os.environ['TOGETHER_API_KEY'])
        stream = together_ai.chat.completions.create(
            model=model,
            messages=messages,
            #temperature=temperature,
            #top_k=top_k,
            #top_p=top_p,
            #max_tokens=max_tokens,
            stream=True
        )

        output = ""
        for chunk in stream:
            content1 = chunk.choices[0].delta.content
            if content1:
                output += content1

        logger.debug(f"Received output from TogetherAI. Output length: {len(output)}")
        return output
    except Exception as e:
        logger.error(f"Error in TogetherAI request: {e}")
        return chunk