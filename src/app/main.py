from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from tempfile import NamedTemporaryFile
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Optional
from groq import Groq
import pandas as pd
import torchaudio
import torch
import asyncio
import logging
import shutil
import uuid
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.data_models import AudioProcessingResponse, ProcessingOptions
from src.lib.whisperx import AudioProcessingPipeline

app = FastAPI()

load_dotenv()  # Ensures environment variables from .env file are loaded

api_key = os.getenv("GROQ_API_KEY", None)
if api_key is None:
    raise EnvironmentError("GROQ_API_KEY is required but not set in environment variables.")
client = Groq(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    device = "cpu"
    compute_type = "float32"
else:
    device = "cuda"
    compute_type = "float16"

start_pipeline = time.time()
print("Loading pipeline...")
pipeline = AudioProcessingPipeline(model_type="base", device=device, compute_type=compute_type)
logger.info(f"Pipeline and model loading done in {time.time() - start_pipeline:.2f} seconds.")

logger.info("Model preloaded at startup.")


async def process_text_with_groq(text: str, prompt: str) -> str:
    logger.info(f"Processing text with Groq: {text[:30]}...")  # Log the first 30 characters
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": prompt + text,
                }
            ],
        )
        result = response.choices[0].message.content
        logger.info(f"Received response from Groq: {result[:30]}...")  # Log the first 30 characters of the response
        return result
    except Exception as e:
        logger.error(f"Error processing text with Groq: {e}")
        raise

def format_diarization_result(diraization_result, transcription_result):
    messages = []
    for segment in diraization_result:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']

        text_segment = ""
        for t_segment in transcription_result['segments']:
            if t_segment['start'] <= start_time and t_segment['end'] >= end_time:
                text_segment = t_segment['text']
                break
        
        messages.append({"role": speaker, "content": text_segment})
    return messages


@app.get("/health-check")
async def health_check():
    return {"status": "Healthy!"}


@app.get("/toy-data")
async def get_toy_data():
    summary = "This is the call summary"
    overall_sentiment = "Positive"
    transcription = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello back"}
    ]
    return {
        "summary": summary,
        "sentiment": overall_sentiment,
        "transcription": transcription
    }


@app.post("/process-audio/", response_model=AudioProcessingResponse)
async def process_audio(file: UploadFile = File(...), 
                        align: Optional[str] = Form(None),
                        diarize: Optional[str] = Form(None),
                        chat_transcription: Optional[str] = Form(None),
                        summarize: Optional[str] = Form(None),
                        analyze_sentiment: Optional[str] = Form(None),
                        extract_keywords: Optional[str] = Form(None),
                        ):
    options = ProcessingOptions(
        align=align.lower() == 'true' if align else False,
        diarize=diarize.lower() == 'true' if diarize else False,
        chat_transcription=chat_transcription.lower() == 'true' if chat_transcription else False,
        summarize=summarize.lower() == 'true' if summarize else False,
        analyze_sentiment=analyze_sentiment.lower() == 'true' if analyze_sentiment else False,
        extract_keywords=extract_keywords.lower() == 'true' if extract_keywords else False,
    )
    tmp_path = None
    logger.info(f"Options received: {options}")
    try:
        start_time = time.time()
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        logger.info(f"File saved to temporary path in {time.time() - start_time:.2f} seconds.")

        transcription_start_time = time.time()
        transcription_result = pipeline.transcribe(tmp_path)
        transcription_time = time.time() - transcription_start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds.")
        logger.info(f"Transcription result: {transcription_result}")

        # Extract the transcribed text from the segments
        transcribed_text = " ".join(segment['text'] for segment in transcription_result['segments'])

        audio_length_seconds = sum(segment['end'] - segment['start'] for segment in transcription_result['segments'])

        response = AudioProcessingResponse(
            filename=file.filename,
            audio_length_seconds=audio_length_seconds,
            transcription_time_seconds=transcription_time,
            transcription=transcribed_text
        )

        if True:
            response.summary = await process_text_with_groq(transcribed_text, "Summarize this text: ")

        if options.align:
            align_start_time = time.time()
            response.alignment = pipeline.align_transcription(transcription_result, tmp_path)
            logger.info(f"Alignment completed in {time.time() - align_start_time:.2f} seconds.")
        if options.diarize:
            diarize_start_time = time.time()
            diarization_result = pipeline.diarize_audio(tmp_path)
            if isinstance(diarization_result, pd.DataFrame):
                diarization_result = diarization_result.to_dict(orient='records')
            response.diarization = diarization_result
            logger.info(f"Diarization completed in {time.time() - diarize_start_time:.2f} seconds.")
        if options.chat_transcription:
            formatted_diarization_result = format_diarization_result(diarization_result, transcription_result)
            response.chat_transcription = formatted_diarization_result
        if options.summarize:
            summarize_start_time = time.time()
            response.summary = await process_text_with_groq(transcribed_text, "Summarize this text: ")
            logger.info(f"Summarization completed in {time.time() - summarize_start_time:.2f} seconds.")
        if options.analyze_sentiment:
            sentiment_start_time = time.time()
            response.sentiment = await process_text_with_groq(transcribed_text, "You are a sentiment analysis model that classifies some text as one of the following: Positive, Neutral, Negative. Your output MUST be the single word that classifies the text, with no explanatory text. Here is the text:")
            logger.info(f"Sentiment analysis completed in {time.time() - sentiment_start_time:.2f} seconds.")
        if options.extract_keywords:
            keywords_start_time = time.time()
            keywords_result = await process_text_with_groq(transcribed_text, "You're job is to extract keywords from a transcription that might be useful to search for later on. Your output should only be the keywords and nothing else. Do not explain anything, simply list the words. Extract keywords from this text: ")
            response.keywords = keywords_result.split()
            logger.info(f"Keyword extraction completed in {time.time() - keywords_start_time:.2f} seconds.")

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        return response

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)