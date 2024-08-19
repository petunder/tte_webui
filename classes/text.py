# classes/text.py
from abc import ABC, abstractmethod
import importlib
import logging
import re
import langid
import os
import subprocess
import json
from llm.providers.together import process_chunk 
from llm.providers.groq import improve_text 


class LLM(ABC):
    @abstractmethod
    def process_chunk(self, chunk, model, system_prompt, temperature, top_k, top_p, repeat_penalty, max_tokens):
        pass

class Text:
    def _initialize_llm(self, provider):
        if provider == "ollama":
            return OllamaLLM()
        elif provider == "together":
            return TogetherLLM()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    def __init__(self, provider="ollama"):
        self.provider = provider
        self.llm = self._initialize_llm(provider)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.language = None

    def detect_language(self, text):
        # Здесь langid определит язык текста
        self.language, confidence = langid.classify(text)
        self.logger.debug(f"Detected language: {self.language} with confidence {confidence}")
        return self.language

    def clean_llm_response(self, text):
        pattern = r'<edited_text>\s*(.*?)\s*</edited_text>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            cleaned_text = ' '.join(matches)
            # Дополнительная очистка от оставшихся тегов
            cleaned_text = cleaned_text.replace('<edited_text>', '').replace('</edited_text>', '')
            final_text = ' '.join(cleaned_text.split())
            self.logger.debug(f"Cleaned text: {final_text[:100]}...")  # Логируем начало очищенного текста
            return final_text
        else:
            self.logger.warning("No <edited_text> tags found in the response.")
            # На всякий случай удаляем теги и из исходного текста
            cleaned_text = text.replace('<edited_text>', '').replace('</edited_text>', '')
            final_text = cleaned_text.strip()
            self.logger.debug(f"Cleaned text (no tags found): {final_text[:100]}...")  # Логируем начало очищенного текста
            return final_text


    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, sentences, max_chunk_size=1000):
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


    def enhance_text(self, text, model, system_prompt, chunk_size=1000):
        self.logger.debug(f"Starting text enhancement. Text length: {len(text)}")
        sentences = self.split_into_sentences(text)
        self.logger.debug(f"Split into {len(sentences)} sentences")
        chunks = self.create_chunks(sentences, chunk_size)
        self.logger.debug(f"Created {len(chunks)} chunks")
        
        edited_chunks = []

        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            edited_chunk = self.llm.process_chunk(
                chunk=chunk,
                model=model,
                system_prompt=system_prompt,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.2,
                max_tokens=2048
            )
            self.logger.debug(f"Chunk {i+1} processed. Original length: {len(chunk)}, Edited length: {len(edited_chunk)}")
            self.logger.debug(f"Edited chunk {i+1}: {edited_chunk[:100]}...")  # Логируем начало отредактированного чанка
            edited_chunks.append(edited_chunk)

        final_edited_text = ' '.join(edited_chunks)
        self.logger.debug(f"All chunks processed. Final text length: {len(final_edited_text)}")
        self.logger.debug(f"Final edited text before cleaning: {final_edited_text[:200]}...")  # Логируем начало финального текста перед очисткой
        cleaned_text = self.clean_llm_response(final_edited_text)
        self.logger.debug(f"Text cleaned. Final cleaned text length: {len(cleaned_text)}")
        self.logger.debug(f"Final cleaned text: {cleaned_text[:200]}...")  # Логируем начало очищенного текста
        return cleaned_text

    def generate_ssml(self, text, model):
        # Обновляем системное приглашение для корректной генерации SSML
        ssml_prompt = "You are an expert in generating SSML (Speech Synthesis Markup Language) for text-to-speech systems. Your task is to take the given text and convert it into SSML format, adding appropriate tags to enhance the speech output. Focus on adding tags for emphasis, pauses, and pronunciation where necessary."
        
        self.logger.debug(f"Generating SSML for text: {text[:100]}...")
        ssml = self.llm.process_chunk(
            chunk=text,
            model=model,
            system_prompt=ssml_prompt,  # Используем ssml_prompt
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.2,
            max_tokens=2048
        )
        self.logger.debug(f"Generated SSML: {ssml[:100]}...")
        return ssml


class OllamaLLM(LLM):
    def process_chunk(self, chunk, model, system_prompt, temperature, top_k, top_p, repeat_penalty, max_tokens):
        logger = logging.getLogger(__name__)
        logger.debug(f"Processing chunk with Ollama. Chunk length: {len(chunk)}")
        try:
            from llm.providers import ollama
            result = ollama.process_chunk(
                chunk=chunk,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens
            )
            logger.debug(f"Chunk processed successfully. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error processing chunk with Ollama: {e}")
            return chunk
class TogetherLLM(LLM):
    def process_chunk(self, chunk, model, system_prompt, temperature, top_k, top_p, repeat_penalty, max_tokens):
        logger = logging.getLogger(__name__)
        logger.debug(f"Processing chunk with Together. Chunk length: {len(chunk)}")
        try:
            # Импортируйте библиотеку или используйте API для работы с Together.
            from llm.providers import together
            result = together.process_chunk(
                chunk=chunk,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens
            )
            logger.debug(f"Chunk processed successfully. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error processing chunk with Together: {e}")
            return chunk

class GROQLLM(LLM):
    def process_chunk(self, chunk, model, system_prompt, temperature, top_k, top_p, repeat_penalty, max_tokens):
        logger = logging.getLogger(__name__)
        logger.debug(f"Processing chunk with Groq. Chunk length: {len(chunk)}")
        try:
            # Импортируйте библиотеку или используйте API для работы с Together.
            from llm.providers import groq
            result = groq.improve_text(
                chunk=chunk,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens
            )
            logger.debug(f"Chunk processed successfully. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error processing chunk with Groq: {e}")
            return chunk

