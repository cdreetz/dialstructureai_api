import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.lib.whisperx import AudioProcessingPipeline

def main():
  pipeline = AudioProcessingPipeline(model_type="base", device="cpu", compute_type="float32")

  audio_file_path = r"/Users/christianreetz/Desktop/call-center/backend_v1/data/4074_30sec.mp3"

  if not os.path.exists(audio_file_path):
    print(f"Audio file not found: {audio_file_path}")
    return

  transcription_result = pipeline.transcribe(audio_file_path)

  print("Transcription Result:")
  print(transcription_result)

if __name__ == "__main__":
  main()