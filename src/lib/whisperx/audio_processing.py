# src/lib/whisperx/audio_processing.py

from .transcribe import load_model
from .alignment import load_align_model, align
from .audio import load_audio
from .diarize import assign_word_speakers, DiarizationPipeline
import os
import gc
import torch



class AudioProcessingPipeline:
    def __init__(self, model_type="large-v2", device="cuda", compute_type="float16", hf_api_key=None):
        # Adjust for available CUDA
        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "float32"
        self.model_type = model_type
        self.device = device
        self.compute_type = compute_type
        self.hf_api_key = hf_api_key

    def transcribe(self, audio_path, batch_size=16):
        model = load_model(self.model_type, self.device, compute_type=self.compute_type)
        audio = load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)
        # Clean up
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        del model
        return result

    def align_transcription(self, transcription_result, audio_path):
        model, metadata = load_align_model(language_code=transcription_result["language"], device=self.device)
        audio = load_audio(audio_path)
        result = align(transcription_result["segments"], model, metadata, audio, self.device, return_char_alignments=False)
        # Clean up
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        del model
        return result

    def diarize_audio(self, audio_path, min_speakers=None, max_speakers=None):
        diarize_model = DiarizationPipeline(use_auth_token=self.hf_api_key, device=self.device)
        audio = load_audio(audio_path)
        segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        return segments

    def assign_speaker_roles(self, diarization_result, transcription_result):
        return assign_word_speakers(diarization_result, transcription_result)
