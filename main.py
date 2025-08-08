from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.model import TurkishLLM
from src.schemas import QueryRequest, QueryResponse, HealthResponse

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
llm_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıcında ve kapanışında çalışacak kodlar"""
    # Startup
    global llm_model
    try:
        logger.info("Model yükleniyor...")
        llm_model = TurkishLLM()
        logger.info("Model başarıyla yüklendi!")
    except Exception as e:
        logger.error(f"Model yüklenemedi: {str(e)}")
        llm_model = None
    
    yield
    
    # Shutdown
    logger.info("Uygulama kapatılıyor...")

# FastAPI uygulaması
app = FastAPI(
    title="Türkçe LLM Test API",
    description="Türkçe GPT-2 modeli ile role-based metin üretimi API'si",
    version="1.0.0",
    lifespan=lifespan
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Ana sayfa"""
    return {
        "message": "Türkçe LLM Test API'sine hoş geldiniz!",
        "docs": "/docs",
        "health": "/health",
        "roles": "/roles"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Sistem durumu kontrolü"""
    if llm_model is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
            message="Model yüklenmedi!"
        )
    
    return HealthResponse(
        status="healthy" if llm_model.is_model_loaded() else "error",
        model_loaded=llm_model.is_model_loaded(),
        message="Sistem çalışıyor!" if llm_model.is_model_loaded() else "Model hatası!"
    )

@app.get("/roles", response_model=dict)
async def get_available_roles():
    """Mevcut roller ve endpoint'leri listele"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi!")
    
    available_roles = llm_model.get_available_roles()
    
    role_descriptions = {
        "sales": "Satış danışmanı - ürün/hizmet önerileri ve satış desteği",
        "custom": "Özel prompt template - basit soru-cevap formatı"
    }
    
    return {
        "available_roles": {
            role: {
                "endpoint": f"/generate/{role}",
                "description": role_descriptions.get(role, "Açıklama bulunamadı")
            }
            for role in available_roles
        },
        "usage": "POST /generate/{role_name} ile kullanın"
    }

# Tek endpoint - role parameter ile
@app.post("/generate/{role}", response_model=QueryResponse)
async def generate_text_with_role(role: str, request: QueryRequest):
    """Role-based metin üretimi endpoint'i"""
    
    # Model kontrolü
    if llm_model is None or not llm_model.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model yüklenmedi veya kullanılamaz durumda!"
        )
    
    # Role kontrolü
    available_roles = llm_model.get_available_roles()
    if role not in available_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz role: {role}. Mevcut roller: {', '.join(available_roles)}"
        )
    
    try:
        # Model'in generate_response metodunu role ile çağır
        response = llm_model.generate_response(
            query=request.query,
            max_length=request.max_length,
            temperature=request.temperature,
            role=role
        )
        
        return QueryResponse(
            query=request.query,
            response=response,
            success=True,
            model_info=f"{llm_model.model_name} ({role.capitalize()} Role)"
        )
        
    except Exception as e:
        logger.error(f"Metin üretilirken hata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metin üretilemedi: {str(e)}"
        )

# Backward compatibility için eski endpoint'ler




@app.post("/generate-custom", response_model=QueryResponse)
async def generate_custom(request: QueryRequest):
    """Özel prompt (backward compatibility)"""
    return await generate_text_with_role("custom", request)

def customize_query_sales_assistant(query: str) -> str:
    """Satış danışmanı için özelleştirilmiş prompt"""
    return f"""Müşteri: {query}

Satış Danışmanı: Merhaba! Size nasıl yardımcı olabilirim?"""

@app.post("/generate-sales-assistant", response_model=QueryResponse)
async def generate_sales_assistant(request: QueryRequest):
    """Satış danışmanı rolü"""
    
    if llm_model is None or not llm_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model yüklenmedi!")
    
    try:
        customized_query = customize_query_sales_assistant(request.query)
        
        response = llm_model.generate_response(
            query=customized_query,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return QueryResponse(
            query=request.query,
            response=response,
            success=True,
            model_info=f"{llm_model.model_name} (Sales Assistant)"
        )
        
    except Exception as e:
        logger.error(f"Satış danışmanı hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metin üretilemedi: {str(e)}")

@app.post("/generate-sales", response_model=QueryResponse)
async def generate_sales_assistant(request: QueryRequest):
    """Satış danışmanı rolü"""
    
    if llm_model is None or not llm_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model yüklenmedi!")
    
    try:
        customized_query = customize_query_sales_assistant(request.query)
        
        response = llm_model.generate_response(
            query=customized_query,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return QueryResponse(
            query=request.query,
            response=response,
            success=True,
            model_info=f"{llm_model.model_name} (Sales Assistant)"
        )
        
    except Exception as e:
        logger.error(f"Satış danışmanı hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metin üretilemedi: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)