from src.models.data_models import AudioProcessingResponse, ProcessingOptions
from src.lib.whisperx.audio_processing import AudioProcessingPipeline
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import torch
import logging
import time
from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)


class AudioProcessingApp:
    def __init__(self):
        load_dotenv()  # Ensures environment variables from .env file are loaded

        self.api_key = os.getenv("GROQ_API_KEY", None)
        if self.api_key is None:
            raise EnvironmentError("GROQ_API_KEY is required but not set in environment variables.")
        
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if self.hf_api_key is None:
            raise EnvironmentError("HUGGINGFACE_API_KEY is required but not set in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        
        self.app = FastAPI()
        self.setup_routes()
        self.pipeline = self.preload_model()

    def preload_model(self):
        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "float32"
        else:
            device = "cuda"
            compute_type = "float16"
        
        pipeline = AudioProcessingPipeline(model_type="base", device=device, compute_type=compute_type, hf_api_key=self.hf_api_key)
        logging.info("Model preloaded at startup.")
        return pipeline

    async def process_text_with_groq(self, text: str, prompt: str) -> str:
        logging.info(f"Processing text with Groq: {text[:30]}...")  # Log the first 30 characters
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": prompt + text,
                    }
                ],
            )
            result = response.choices[0].message.content
            logging.info(f"Received response from Groq: {result[:30]}...")  # Log the first 30 characters of the response
            return result
        except Exception as e:
            logging.error(f"Error processing text with Groq: {e}")
            raise

    def format_diarization_result(self, diarization_result, transcription_result):
        messages = []
        for segment in diarization_result:
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']

            text_segments = []
            for t_segment in transcription_result['segments']:
                if t_segment['end'] > start_time and t_segment['start'] < end_time:
                    text_segments.append(t_segment['text'])
            
            text_segment = " ".join(text_segments).strip()
            messages.append({"role": speaker, "content": text_segment})
        return messages

    def setup_routes(self):
        @self.app.get("/health-check")
        async def health_check():
            return {"status": "Healthy!"}

        @self.app.get("/toy-data")
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

        @self.app.post("/process-audio/", response_model=AudioProcessingResponse)
        async def process_audio(file: UploadFile = File(...), 
                                align: Optional[str] = Form(None),
                                diarize: Optional[str] = Form(None),
                                chat_transcription: Optional[str] = Form(None),
                                summarize: Optional[str] = Form(None),
                                analyze_sentiment: Optional[str] = Form(None),
                                extract_keywords: Optional[str] = Form(None)):
            options = ProcessingOptions(
                align=align.lower() == 'true' if align else False,
                diarize=diarize.lower() == 'true' if diarize else False,
                chat_transcription=chat_transcription.lower() == 'true' if chat_transcription else False,
                summarize=summarize.lower() == 'true' if summarize else False,
                analyze_sentiment=analyze_sentiment.lower() == 'true' if analyze_sentiment else False,
                extract_keywords=extract_keywords.lower() == 'true' if extract_keywords else False,
            )
            tmp_path = None
            try:
                start_time = time.time()
                with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp_path = tmp.name
                    shutil.copyfileobj(file.file, tmp)
                logging.info(f"File saved to temporary path in {time.time() - start_time:.2f} seconds.")

                transcription_result = None
                aligned_transcription_result = None
                diarization_result = None
                speaker_assigned_result = None

                # Transcription
                transcription_start_time = time.time()
                transcription_result = self.pipeline.transcribe(tmp_path)
                logging.debug(f"Transcription Result: {transcription_result}")

                transcription_time = time.time() - transcription_start_time
                logging.info(f"Transcription completed in {transcription_time:.2f} seconds.")
                logging.info(f"Transcription result: {transcription_result}")

                if options.align:
                    # Alignment
                    align_start_time = time.time()
                    aligned_transcription_result = self.pipeline.align_transcription(transcription_result, tmp_path)
                    logging.info(f"Alignment completed in {time.time() - align_start_time:.2f} seconds.")
                    logging.info(f"Aligned transcription result: {aligned_transcription_result}")

                if options.diarize:
                    # Diarization
                    diarize_start_time = time.time()
                    diarization_result = self.pipeline.diarize_audio(tmp_path)
                    logging.info(f"Diarization completed in {time.time() - diarize_start_time:.2f} seconds.")
                    logging.info(f"Diarization result: {diarization_result}")

                if options.chat_transcription:
                    if not aligned_transcription_result:
                        aligned_transcription_result = self.pipeline.align_transcription(transcription_result, tmp_path)
                    speaker_assigned_result = self.pipeline.assign_speaker_roles(diarization_result, aligned_transcription_result)
                    logging.info(f"Speaker roles assigned.")

                # Extract the transcribed text from the segments
                transcribed_text = " ".join(segment['text'] for segment in speaker_assigned_result['segments']) if options.chat_transcription else (
                    " ".join(segment['text'] for segment in aligned_transcription_result['segments']) if aligned_transcription_result else (
                        " ".join(segment['text'] for segment in transcription_result['segments']) if transcription_result else ""
                    )
                )

                #audio_length_seconds = sum(segment['end'] - segment['start'] for segment in transcription_result['segments'])
                audio_length_seconds = 99.99

                response = AudioProcessingResponse(
                    filename=file.filename,
                    audio_length_seconds=audio_length_seconds,
                    transcription_time_seconds=transcription_time,
                    transcription=transcribed_text
                )

                if options.align:
                    response.alignment = aligned_transcription_result
                if options.diarize:
                    response.diarization = diarization_result
                    if options.chat_transcription:
                        response.chat_transcription = self.format_diarization_result(diarization_result, speaker_assigned_result)
                if options.summarize:
                    logging.info("Summarization option is enabled.")
                    summarize_start_time = time.time()
                    try:
                        response.summary = await self.process_text_with_groq(transcribed_text, "Summarize this text: ")
                        logging.info(f"Summarization completed in {time.time() - summarize_start_time:.2f} seconds.")
                    except Exception as e:
                        logging.error(f"Error during summarization: {e}")
                        response.summary = "Summarization failed."
                if options.analyze_sentiment:
                    sentiment_start_time = time.time()
                    response.sentiment = await self.process_text_with_groq(transcribed_text, "You are a sentiment analysis model that classifies some text as one of the following: Positive, Neutral, Negative. Your output MUST be the single word that classifies the text, with no explanatory text. Here is the text:")
                    logging.info(f"Sentiment analysis completed in {time.time() - sentiment_start_time:.2f} seconds.")
                if options.extract_keywords:
                    keywords_start_time = time.time()
                    keywords_result = await self.process_text_with_groq(transcribed_text, "Extract keywords from this text: ")
                    response.keywords = keywords_result.split()
                    logging.info(f"Keyword extraction completed in {time.time() - keywords_start_time:.2f} seconds.")

                total_time = time.time() - start_time
                logging.info(f"Total processing time: {total_time:.2f} seconds.")
                return response

            except Exception as e:
                logging.error(f"Error processing audio: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

def create_app() -> FastAPI:
    app_instance = AudioProcessingApp()
    return app_instance.app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8010)
