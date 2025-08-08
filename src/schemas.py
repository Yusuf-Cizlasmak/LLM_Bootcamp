from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    """API'ye gönderilen query isteği"""
    query: str = Field(..., description="Modele gönderilecek metin", min_length=1)
    max_length: Optional[int] = Field(150, description="Maksimum cevap uzunluğu", ge=50, le=512)
    temperature: Optional[float] = Field(0.7, description="Yaratıcılık seviyesi (0.1-1.0)", ge=0.1, le=1.0)

class QueryResponse(BaseModel):
    """API'den dönen cevap"""
    query: str = Field(..., description="Gönderilen orijinal query")
    response: str = Field(..., description="Model tarafından üretilen cevap")
    success: bool = Field(..., description="İşlem başarılı mı?")
    model_info: str = Field(..., description="Kullanılan model bilgisi")

class HealthResponse(BaseModel):
    """Sistem durumu cevabı"""
    status: str = Field(..., description="Sistem durumu")
    model_loaded: bool = Field(..., description="Model yüklendi mi?")
    message: str = Field(..., description="Durum mesajı")