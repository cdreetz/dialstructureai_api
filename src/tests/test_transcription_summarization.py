import requests

# URL of the FastAPI endpoint
url = "https://7a91-2600-1700-7b00-5e10-a128-b54f-abd9-33aa.ngrok-free.app/process-audio/"

# Path to the audio file you want to process
audio_file_path = r"/Users/christianreetz/Desktop/call-center/backend_v1/data/4074_30sec.mp3"

# Processing options as individual form fields
data = {
    "align": "false",
    "diarize": "true",
    "chat_transcription": "true",
    "summarize": "true",
    "analyze_sentiment": "true",
    "extract_keywords": "true"
}

# Send POST request
with open(audio_file_path, "rb") as audio_file:
    files = {"file": (audio_file.name, audio_file, "audio/mp3")}
    response = requests.post(url, files=files, data=data)

# Print the response
print(response.json())