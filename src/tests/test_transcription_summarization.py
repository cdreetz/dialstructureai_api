import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8005/process-audio/"

# Path to the audio file you want to process
audio_file_path = r"/Users/christianreetz/Desktop/call-center/backend_v1/data/4074_30sec.mp3"

# Processing options as individual form fields
data = {
    "align": "false",
    "diarize": "false",
    "summarize": "true",
    "analyze_sentiment": "false",
    "extract_keywords": "false"
}

# Send POST request
with open(audio_file_path, "rb") as audio_file:
    files = {"file": (audio_file.name, audio_file, "audio/mp3")}
    response = requests.post(url, files=files, data=data)

# Print the response
print(response.json())