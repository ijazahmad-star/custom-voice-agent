import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.config import VoiceAgentConfig
from core.agent import VoiceAgent

app = FastAPI(title="Voice Agent API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Voice Agent
# Note: This will load models on startup
agent = VoiceAgent(config=VoiceAgentConfig)

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    thread_id: str = Form("web_session_default")
):
    """
    Receives an audio file, processes it, and returns the transcription + response audio.
    """
    try:
        # 1. Save uploaded file
        session_id = str(uuid.uuid4())
        input_filename = os.path.join(TEMP_DIR, f"input_{session_id}.wav")
        output_filename = os.path.join(TEMP_DIR, f"output_{session_id}.wav")
        
        with open(input_filename, "wb") as f:
            f.write(await audio.read())
            
        # 2. Process: STT
        user_text = agent.speech_to_text(input_filename)
        if not user_text:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech detected in audio."}
            )
            
        # 3. Process: LLM
        ai_response = agent.get_llm_response(user_text, thread_id=thread_id)
        
        # 4. Process: TTS
        audio_path = agent.text_to_speech(ai_response, output_filename)
        
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate response audio.")
            
        # 5. Read output audio and encode to base64
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            
        # 6. Cleanup (optional, or keep for history)
        # os.remove(input_filename)
        # os.remove(output_filename)
        
        return {
            "user_text": user_text,
            "ai_response": ai_response,
            "audio_base64": audio_base64,
            "format": "wav"
        }
        
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF file, extracts text, and adds it to the agent's knowledge base.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        import pypdf
        from io import BytesIO
        
        # 1. Read PDF content
        content = await file.read()
        pdf_file = BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)
        
        # 2. Extract text from all pages
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")
            
        # 3. Add to VoiceAgent Knowledge Base
        success = agent.add_knowledge(text)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update knowledge base.")
            
        return {
            "filename": file.filename,
            "status": "success",
            "message": f"Successfully indexed {len(text)} characters from {len(reader.pages)} pages."
        }
        
    except Exception as e:
        print(f"PDF Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
