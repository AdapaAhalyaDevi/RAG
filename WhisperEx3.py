#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openai
import os
os.environ["OpenAI_API_KEY"]="your api key should be given here"
openai.api_key = os.getenv("OpenAI_API_Key")

audio_file = open(r"C:\Users\ahalyapc\Downloads\Conversation between Teacher and Student _ Conversation in the classroom _ Adrija Biswas.mp4", "rb")
transcription = openai.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)

response = openai.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
        { 'role':'system','content':'Give conversation between teacher and students'},
        {'role':'user','content':f" Give conversation between teacher and students in bullet point form and specify it whether the teacher is speaking or student is speaking and the teacher was calling the roll no's then the students are responding:\n{transcription.text}"}
    ]
)
print(response.choices[0].message.content)

