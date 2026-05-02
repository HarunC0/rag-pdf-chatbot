import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()



def pdf_oku(pdf):
    metin = ""
    reader = PdfReader(pdf)
    for sayfa in reader.pages:
        metin += sayfa.extract_text()
    return metin

def vektor_db_olustur(metin):
    parcalayici = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parcalar = parcalayici.split_text(metin)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_texts(parcalar, embeddings)

def soru_sor(soru, vektor_db, sohbet_gecmisi):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    benzer_belgeler = vektor_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([b.page_content for b in benzer_belgeler])
    gecmis_metin = "\n".join([f"{m['rol']}: {m['icerik']}" for m in sohbet_gecmisi[-6:]])
    prompt = ChatPromptTemplate.from_template("""
Asagidaki belge parcalarina ve sohbet gecmisine dayanarak soruyu Turkce cevapla.

Belge:
{baglam}

Sohbet gecmisi:
{gecmis}

Soru: {soru}

Cevap:""")
    zincir = prompt | llm
    cevap = zincir.invoke({"baglam": baglam, "gecmis": gecmis_metin, "soru": soru})
    return cevap.content

st.title("PDF'e Soru Sor")
st.write("Bir PDF yukle, istedigin soruyu sor.")

pdf = st.file_uploader("PDF yukle", type="pdf")

if pdf:
    if "vektor_db" not in st.session_state:
        with st.spinner("PDF okunuyor..."):
            metin = pdf_oku(pdf)
            st.session_state.vektor_db = vektor_db_olustur(metin)
            st.success("Hazir! Sorularini sorabilirsin.")

    if "gecmis" not in st.session_state:
        st.session_state.gecmis = []

    for mesaj in st.session_state.gecmis:
        with st.chat_message(mesaj["rol"]):
            st.write(mesaj["icerik"])

    soru = st.chat_input("Sorunuzu yazin...")

    if soru:
        with st.chat_message("user"):
            st.write(soru)
        with st.spinner("Dusunuyor..."):
            cevap = soru_sor(soru, st.session_state.vektor_db, st.session_state.gecmis)
        with st.chat_message("assistant"):
            st.write(cevap)
        st.session_state.gecmis.append({"rol": "user", "icerik": soru})
        st.session_state.gecmis.append({"rol": "assistant", "icerik": cevap})
