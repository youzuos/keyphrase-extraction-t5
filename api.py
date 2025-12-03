import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.models import T5KeyphraseModel
import uvicorn
''
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


app = FastAPI(
    title="Keyphrase Extraction API",
    description="T5-based keyphrase extraction service",
    version="1.0.0"
)

model: Optional[T5KeyphraseModel] = None


class TextInput(BaseModel):
    text: str
    max_keywords: Optional[int] = 10


class KeywordResponse(BaseModel):
    keywords: List[str]
    count: int
    text_length: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model(model_path: str = "models/checkpoints/final_model"):
    global model
    if model is None:
        print(f"Loading model: {model_path}")
        model = T5KeyphraseModel()
        try:
            model.load_model(model_path)
            print("success")
        except Exception as e:
            print(f"failed: {e}")
            raise
    return model


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["Root"])
async def root():
    return FileResponse("static/index.html")

@app.get("/api", tags=["Root"])
async def api_info():
    return {
        "message": "Keyphrase Extraction API Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/extract", response_model=KeywordResponse, tags=["Extraction"])
async def extract_keywords(input_data: TextInput):
    try:
        if model is None:
            load_model()
        
        if not input_data.text or not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        keywords = model.predict(input_data.text)
        
        if input_data.max_keywords:
            keywords = keywords[:input_data.max_keywords]
        
        return KeywordResponse(
            keywords=keywords,
            count=len(keywords),
            text_length=len(input_data.text)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/extract/batch", tags=["Extraction"])
async def extract_keywords_batch(texts: List[str]):
    try:
        if model is None:
            load_model()
        
        if not texts:
            raise HTTPException(status_code=400, detail="Text list cannot be empty")
        
        all_keywords = model.predict_batch(texts)
        
        results = []
        for text, keywords in zip(texts, all_keywords):
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "keywords": keywords,
                "count": len(keywords)
            })
        
        return {
            "results": results,
            "total": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    model_path = "models/checkpoints/final_model"
    if Path(model_path).exists():
        try:
            load_model(model_path)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Starting Keyphrase Extraction API Service\n")
    print("Access URLs:")
    print("Frontend: http://localhost:8000/")
    print("API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
