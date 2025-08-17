from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import torch
import json
import nltk

from main import ChatbotAssistant, get_stocks   # import your chatbot classes

# Load trained model
assistant = ChatbotAssistant("intents.json", function_mappings={"stocks": get_stocks})
assistant.parse_intents()
assistant.load_model("chatbot_model.pth", "dimensions.json")

# FastAPI app
app = FastAPI()

# Allow frontend (HTML) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    reply = assistant.process_message(req.message)
    return {"reply": reply}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
