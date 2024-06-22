pip install --upgrade openai

import openai
import os
os.environ["OpenAI_API_KEY"]= "your api key should be given here"
openai.api_key = os.getenv("OpenAI_API_Key")

audio_file = open(r"C:\Users\ahalyapc\Downloads\Warren Buffett On Exposing Business Frauds And Deception.mp3", "rb")
transcription = openai.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)

response = openai.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
        { 'role':'system','content':'You a good at creating bullet point summaries and have knowledge of Warren Buffet'},
        {'role':'user','content':f"Summarize the following in bullet point form:\n{transcription.text}"}
    ]
)
print(response.choices[0].message.content)
