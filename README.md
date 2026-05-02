# PDF'e Soru Sor - RAG Chatbot

PDF dosyalarına yapay zeka ile soru sorabildiğiniz bir chatbot uygulaması.

## Nasıl Çalışır?
1. PDF yükle
2. Soru sor
3. Yapay zeka PDF'i okuyup cevap verir

## Kullanılan Teknolojiler
- Python
- Streamlit
- LangChain
- Groq (Llama 3.3 70B)
- FAISS
- HuggingFace Embeddings

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

.env dosyası oluştur:

Uygulamayı çalıştır:
```bash
streamlit run app.py
```
