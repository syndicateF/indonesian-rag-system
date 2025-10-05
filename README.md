<h1 align="center">Indonesia RAG System</h1>

<p align="center">
  <img src="https://raster.shields.io/badge/Python-3.8%2B-blue.png?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <br/>
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen?logo=githubactions" />
  <img src="https://img.shields.io/badge/Platform-Linux%20|%20Windows-lightgrey?logo=linux" />
  <br/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/SentenceTransformers-Embeddings-blueviolet?logo=python&logoColor=white" />
  <br/>
  <img src="https://img.shields.io/badge/Chroma-Vector%20DB-orange?logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP%20Indonesia-Community-red?logo=github&logoColor=white" />
</p>




Project ini dibuat untuk pembelajaran pribadi menganai Retrieval-Augmented Generation (RAG) untuk Bahasa Indonesia yang menyediakan jawaban akurat berdasarkan dokumen teks berbahasa Indonesia. Sistem ini menggabungkan teknologi embedding modern dengan model bahasa Indonesia untuk memberikan respons yang kontekstual dan relevan.


<h3 align="left">1. Instalasi</h3>

```bash
git clone https://github.com/syndicatef/indonesian-rag-system.git
cd indonesian-rag-system
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

<h3 align="left">2. Command Line Interface</h3>


1. Build Index dari Dokumen

```bash
# Build index dari directory dokumen
python -m src.main --data-dir path/to/your/documents

# Contoh dengan data sampel
python -m src.main --data-dir data/raw/
```

2. Query single

```bash
# Menggunakan model QA (default)
python -m src.main --query "Apa itu kecerdasan buatan?"

# Menggunakan model generative
python -m src.main --query "Jelaskan perkembangan AI" --model-type generative
```

3. query loop
```bash
# Mode interaktif dengan model QA
python -m src.main --interactive

# Mode interaktif dengan model generative
python -m src.main --model-type generative --interactive
```

<h3 align="left">3. Troubleshooting</h3>

1. Model tidak terdownload

```bash
# Login ke Hugging Face pake CLI
huggingface-cli login
```
2. Vector store error
```bash
# Clear dan rebuild vector store
rm -rf data/processed/vector_db
python -m src.main --data-dir data/raw/
```

---


<h3 align="center">📬 Contact Me</h3>

<p align="center">
  <a href="https://instagram.com/faaaajriiiiin">
    <img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?&logo=instagram&logoColor=white" />
  </a>
  <a href="https://wa.me/62895352933680">
    <img src="https://img.shields.io/badge/WhatsApp-25D366?logo=whatsapp&logoColor=white" />
  </a>
  <a href="mailto:fajrintrk313@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white" />
  </a>
  <a href="https://x.com/darkobunny_">
    <img src="https://img.shields.io/badge/X%20(Twitter)-000000?logo=x&logoColor=white" />
  </a>
  <a href="https://tiktok.com/@syndicatef">
    <img src="https://img.shields.io/badge/TikTok-010101?logo=tiktok&logoColor=white" />
  </a>
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  </a>
</p>

<p align="center" style="font-size:12px; opacity:0.7;">
  © 2025 Darko Bunny. All rights reserved.
</p>
