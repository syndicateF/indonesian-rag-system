from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    pipeline
)
from typing import List, Dict, Any
import torch

from src.utils.logger import app_logger

class IndonesianGenerator:
    def __init__(self, model_config, use_tqdm: bool = True):
        self.model_config = model_config
        self.use_tqdm = use_tqdm
        self.model_type = model_config.model_type
        
        app_logger.info(f"Loading {self.model_type} model...")
        
        try:
            if self.model_type == "qa":
                # Load QA model (IndoBERT-lite-squad)
                self.tokenizer = AutoTokenizer.from_pretrained(model_config.qa_model_name)
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_config.qa_model_name)
                app_logger.info(f"Loaded QA model: {model_config.qa_model_name}")
                
            else:
                # Load generative model
                self.tokenizer = AutoTokenizer.from_pretrained(model_config.generative_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_config.generative_model_name)
                
                # Tambahkan padding token untuk GPT2
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                app_logger.info(f"Loaded generative model: {model_config.generative_model_name}")
                
            self.model.eval()
            
        except Exception as e:
            app_logger.error(f"Error loading model: {str(e)}")
            # Fallback ke model QA jika generative model gagal
            app_logger.info("Falling back to QA model...")
            self.model_type = "qa"
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.qa_model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_config.qa_model_name)
    
    def _generate_comprehensive_answer(self, question: str, context: str) -> str:
        """Fallback method untuk jawaban yang lebih komprehensif"""
        try:
            # Approach alternatif: gunakan chunking yang lebih kecil
            sentences = context.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in question.lower().split()):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return ". ".join(relevant_sentences[:3]) + "."
            else:
                return "Informasi yang relevan tidak ditemukan dalam konteks yang diberikan."
                
        except Exception as e:
            app_logger.error(f"Error in comprehensive answer: {str(e)}")
            return "Tidak dapat menghasilkan jawaban yang lengkap."
        
    def generate_qa_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer menggunakan model QA (IndoBERT-lite-squad)"""
        try:
            # Tokenize question and context - PERBAIKAN: hilangkan parameter bermasalah
            inputs = self.tokenizer(
                question, 
                context, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
                # HAPUS: stride=128 dan return_overflowing_tokens=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # Get the most likely answer span
                start_idx = torch.argmax(start_logits, dim=1).item()
                end_idx = torch.argmax(end_logits, dim=1).item() + 1
                
                # Convert tokens back to text
                answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Calculate confidence score
                start_score = torch.max(torch.softmax(start_logits, dim=-1)).item()
                end_score = torch.max(torch.softmax(end_logits, dim=-1)).item()
                confidence = (start_score + end_score) / 2
                
                # Jika jawaban terlalu pendek atau kosong
                if len(answer.strip()) < 2:
                    answer = self._find_best_sentence_answer(question, context)
                    confidence = confidence * 0.8  # Reduce confidence for fallback
                
            return {
                "answer": answer.strip(),
                "confidence": confidence,
                "start_score": start_score,
                "end_score": end_score
            }
            
        except Exception as e:
            app_logger.error(f"Error in QA generation: {str(e)}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam menghasilkan jawaban.",
                "confidence": 0.0
            }
    
    def _find_best_sentence_answer(self, question: str, context: str) -> str:
        """Fallback method untuk mencari jawaban berdasarkan kalimat terbaik"""
        try:
            sentences = context.split('. ')
            question_keywords = [word.lower() for word in question.split() if len(word) > 2]
            
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                score = sum(1 for keyword in question_keywords if keyword in sentence.lower())
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            return best_sentence if best_sentence else sentences[0] + "." if sentences else "Jawaban tidak ditemukan."
            
        except Exception as e:
            app_logger.error(f"Error in best sentence search: {str(e)}")
            return "Tidak dapat menemukan jawaban yang spesifik."
    
    def generate_text_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer menggunakan model generative dengan prompt yang lebih baik"""
        try:
            # Improved prompt dengan instruksi yang lebih jelas
            prompt = f"""Anda adalah asisten AI yang membantu menjawab pertanyaan. 
    Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan.

    KONTEKS:
    {context}

    PERTANYAAN: {question}

    INSTRUKSI:
    - Jawab dalam Bahasa Indonesia yang baik dan benar
    - Berdasarkan hanya pada informasi di konteks
    - Jika informasi tidak cukup, jelaskan secara singkat
    - Jangan menambahkan informasi dari luar konteks

    JAWABAN:"""
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=400,  # Kurangi sedikit untuk fokus yang lebih baik
                    temperature=0.3,  # Kurangi temperature untuk output lebih deterministik
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part (remove the prompt)
            answer = generated_text.replace(prompt, "").strip()
            
            # Clean up the answer
            if "JAWABAN:" in answer:
                answer = answer.split("JAWABAN:")[-1].strip()
            
            return {
                "answer": answer,
                "confidence": 0.7,
                "full_generation": generated_text
            }
            
        except Exception as e:
            app_logger.error(f"Error in text generation: {str(e)}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam menghasilkan jawaban.",
                "confidence": 0.0
            }
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main generation method dengan context dari retrieval"""
        app_logger.info(f"Generating response for query: '{query}'")
        
        # Combine context from retrieved documents
        context_text = "\n\n".join([doc['content'] for doc in context])
        
        if not context_text.strip():
            return {
                "answer": "Maaf, tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda.",
                "confidence": 0.0,
                "sources": []
            }
        
        try:
            if self.model_type == "qa":
                result = self.generate_qa_answer(query, context_text)
            else:
                result = self.generate_text_answer(query, context_text)
            
            # Add source information
            result["sources"] = [
                {
                    "source": doc.get('source', 'unknown'),
                    "content_preview": doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                }
                for doc in context
            ]
            
            app_logger.info(f"Successfully generated response with confidence: {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            app_logger.error(f"Error in generation pipeline: {str(e)}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "confidence": 0.0,
                "sources": []
            }