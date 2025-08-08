# src/model.py
from llama_cpp import Llama
import logging
import os
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurkishLLM:
    def __init__(self):
        """Turkish Llama 8B model initialization"""
        self.model = None
        self.model_name = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1-GGUF"
        
        # Inference parameters
        self.inference_params = {
            "n_threads": 4,
            "n_predict": -1,
            "top_k": 40,
            "min_p": 0.05,
            "top_p": 0.95,
            "temp": 0.8,
            "repeat_penalty": 1.1,
            "input_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
            "input_suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "antiprompt": [],
            "pre_prompt": "Sen bir yapay zeka asistanısın. Kullanıcı sana bir görev verecek. Amacın görevi olabildiğince sadık bir şekilde tamamlamak.",
            "pre_prompt_suffix": "<|eot_id|>",
            "pre_prompt_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "seed": -1,
            "tfs_z": 1,
            "typical_p": 1,
            "repeat_last_n": 64,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n_keep": 0,
            "logit_bias": {},
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "memory_f16": True,
            "multiline_input": False,
            "penalize_nl": True
        }
        
        # Role-based system prompts
        self.role_prompts = {
            "turkcell": "Sen Turkcell müşteri hizmetleri asistanısın. Kibar, profesyonel ve çözüm odaklı ol. Müşteri memnuniyetini ön planda tut.",
            "support": "Sen yardımsever bir destek asistanısın. Kullanıcıların sorunlarına pratik ve anlaşılır çözümler sun.",
            "tech": "Sen teknik destek uzmanısın. Teknik sorunları adım adım, anlaşılır bir dille çöz.",
            "sales": "Sen satış danışmanısın. Ürün ve hizmetler hakkında bilgilendirici ol, müşteri ihtiyaçlarına uygun öneriler sun.",
            "custom": "Sen yardımcı bir yapay zeka asistanısın.",
            "default": self.inference_params["pre_prompt"]
        }
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _load_model(self):
        """Load the Llama model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Model parametreleri
            model_kwargs = {
                "repo_id": self.model_name,
                "filename": "*Q4_K.gguf",
                "verbose": False,
                "n_ctx": 2048,  # Context window
                "n_batch": 512,  # Batch size for prompt processing
                "n_threads": os.cpu_count() or 4,  # CPU threads
            }
            
            # GPU varsa kullan
            try:
                import torch
                if torch.cuda.is_available():
                    model_kwargs["n_gpu_layers"] = -1  # Tüm layerları GPU'ya yükle
                    logger.info("GPU detected, using CUDA acceleration")
            except ImportError:
                logger.info("No GPU/CUDA support, using CPU")
            
            self.model = Llama.from_pretrained(**model_kwargs)
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_available_roles(self) -> list:
        """Return available roles"""
        return list(self.role_prompts.keys())

    def _construct_prompt(self, query: str, role: str = "default") -> str:
        """Construct prompt based on role"""
        system_prompt = self.role_prompts.get(role, self.role_prompts["default"])
        
        prompt = (
            f"{self.inference_params['pre_prompt_prefix']}"
            f"{system_prompt}"
            f"{self.inference_params['pre_prompt_suffix']}"
            f"{self.inference_params['input_prefix']}"
            f"{query}"
            f"{self.inference_params['input_suffix']}"
        )
        
        return prompt

    def generate(self, text: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate text using the base model"""
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded")
        
        try:
            # Simple generation without role
            prompt = self._construct_prompt(text, role="default")
            
            response = self.model(
                prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=self.inference_params["top_p"],
                top_k=self.inference_params["top_k"],
                repeat_penalty=self.inference_params["repeat_penalty"],
                stop=["<|eot_id|>", "<|end_of_text|>"]
            )
            
            generated_text = response['choices'][0]['text'].strip()
            return generated_text if generated_text else "Yanıt üretilemedi."
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Üretim hatası: {str(e)}"

    def generate_response(self, query: str, max_length: int = 256, 
                         temperature: float = 0.7, role: str = "default") -> str:
        """Generate response based on role"""
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded")
        
        try:
            # Construct role-based prompt
            prompt = self._construct_prompt(query, role)
            
            # Log for debugging
            logger.debug(f"Using role: {role}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=self.inference_params["top_p"],
                top_k=self.inference_params["top_k"],
                repeat_penalty=self.inference_params["repeat_penalty"],
                stop=["<|eot_id|>", "<|end_of_text|>", "\n\nUser:", "\n\nMüşteri:"]
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean up response
            generated_text = generated_text.replace("<|eot_id|>", "").strip()
            
            return generated_text if generated_text else "Yanıt üretilemedi."
            
        except Exception as e:
            logger.error(f"Generation error for role {role}: {e}")
            return f"Üretim hatası: {str(e)}"

# Test code
if __name__ == "__main__":
    try:
        print("Turkish Llama 8B Model Test")
        print("-" * 50)
        
        llm = TurkishLLM()
        
        if llm.is_model_loaded():
            print("✓ Model başarıyla yüklendi!")
            print(f"✓ Mevcut roller: {llm.get_available_roles()}")
            print("-" * 50)
            
            # Test different roles
            test_queries = [
                ("Türkiye'nin başkenti neresidir?", "default"),
                ("İnternet bağlantım yavaş", "tech"),
                ("Yeni telefon tarifeleriniz hakkında bilgi alabilir miyim?", "turkcell"),
            ]
            
            for query, role in test_queries:
                print(f"\n[{role.upper()}] Soru: {query}")
                response = llm.generate_response(query, role=role)
                print(f"Yanıt: {response}")
                print("-" * 50)
        else:
            print("✗ Model yüklenemedi!")
            
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()