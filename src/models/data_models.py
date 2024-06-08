from pydantic import BaseModel
from typing import List, Optional, Dict

class TranscriptionRequest(BaseModel):
    file_name: str
    content_type: str

class TranscriptionResponse(BaseModel):
    transcription: str

class SummaryResponse(BaseModel):
    summary: str

class SentimentResponse(BaseModel):
    sentiment: str


class KeywordsResponse(BaseModel):
    keywords: List[str]

class AudioProcessingResponse(BaseModel):
    filename: str
    audio_length_seconds: float
    transcription_time_seconds: float
    transcription: str
    alignment: Optional[str] = None
    diarization: Optional[List[Dict[str, str]]] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    keywords: Optional[List[str]] = None
    chat_transcription: Optional[List[Dict[str, str]]] = None
    file_details: Optional[str] = None

class ProcessingOptions(BaseModel):
    align: bool = False
    diarize: bool = False
    summarize: bool = False
    analyze_sentiment: bool = False
    extract_keywords: bool = False
