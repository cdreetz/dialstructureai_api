from .transcribe import load_model
from .alignment import load_align_model, align
from .audio import load_audio
from .diarize import assign_word_speakers, DiarizationPipeline
from dotenv import load_dotenv
import os
import gc
import torch

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


class AudioProcessingPipeline:
    def __init__(self, model_type="large-v2", device="cuda", compute_type="float16"):
      if not torch.cuda.is_available():
        device = "cpu"
        compute_type = "float32"
      self.model_type = model_type
      self.device = device
      self.compute_type = compute_type

    def transcribe(self, audio_path, batch_size=16):
        model = load_model(self.model_type, self.device, compute_type=self.compute_type)
        audio = load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)
        gc.collect()
        if self.device == "cuda":
          torch.cuda.empty_cache()
        del model
        return result

    def align_transcription(self, transcription_result, audio_path):
        model, metadata = load_align_model(language_code=transcription_result["language"], device=self.device)
        audio = load_audio(audio_path)
        result = align(transcription_result["segments"], model, metadata, audio, self.device, return_char_alignments=False)
        gc.collect()
        if self.device == "cuda":
          torch.cuda.empty_cache()
        del model
        return result

    def diarize_audio(self, audio_path, use_auth_token=HUGGINGFACE_API_KEY, min_speakers=None, max_speakers=None):
        diarize_model = DiarizationPipeline(use_auth_token=use_auth_token, device=self.device)
        audio = load_audio(audio_path)
        segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        return segments
