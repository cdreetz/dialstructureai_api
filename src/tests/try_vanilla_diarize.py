from pyannote.audio import Pipeline
import torchaudio

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=""
)

audio_path = r"/Users/christianreetz/Desktop/call-center/backend_v1/data/4074_30sec.mp3"
audio_data, sample_rate = torchaudio.load(audio_path)
diarization = pipeline({"waveform": audio_data, "sample_rate": sample_rate})

with open("audio.rttm", "w") as rttm:
  diarization.write_rttm(rttm)