
import streamlit as st
import sounddevice as sd
import librosa
import numpy as np
import pickle
from collections import Counter, defaultdict
import os
from google.cloud import speech
import queue
import matplotlib.pyplot as plt
from textblob import TextBlob  # Duygu analizi için

# JSON dosyasının yolunu belirtin
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speech_to_text_key.json"

# Eğitilmiş modeli yükle
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Konuşmacı isimleri
class_names = ["Dilay", "Ümit"]

# Google Speech-to-Text API kullanarak ses verisini metne dönüştür
def transcribe_audio(audio_data, samplerate):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code="tr-TR"
    )
    response = client.recognize(config=config, audio=audio)
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    return transcript

# Özellik çıkarımı ve model tahmini
def process_audio(indata, samplerate):
    mfcc = librosa.feature.mfcc(y=indata.flatten(), sr=samplerate, n_mfcc=20).T
    delta_mfcc = librosa.feature.delta(mfcc)
    features = np.hstack((mfcc, delta_mfcc))
    predictions = svm.predict(features)
    speaker = Counter(predictions).most_common(1)[0][0]  # Çoğunluk oyu
    return class_names[speaker]

# Duygu tahmini
def analyze_emotion(text):
    # Önce metni küçük harflere çevir ve noktalama işaretlerini kaldır
    text = text.lower().strip()
    if len(text.split()) < 3:  # Çok kısa metinlerde duygu analizi yerine nötr döndür
        return "Nötr", 0

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Mutlu", polarity * 100
    elif polarity < 0:
        return "Üzgün", abs(polarity) * 100
    else:
        return "Nötr", 100

# Konu tahmini (örnek)
def analyze_topic(text):
    categories = {
        "Spor": ["spor", "maç", "futbol", "basketbol", "voleybol"],
        "Teknoloji": ["teknoloji", "bilgisayar", "yazılım", "internet", "uygulama"],
        "Dünya": ["dünya", "haber", "uluslararası", "siyaset", "ekonomi"],
        "Sanat": ["sanat", "resim", "müzik", "tiyatro", "edebiyat"]
    }

    text = text.lower()
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Diğer"

# Dinamik alanlar
st.title("Konuşmacı Tanımlama, Kelime Sayımı, Duygu ve Konu Analizi")
st.write("Bu uygulama, konuşmacıyı tanımlar, kelimelerini sayar, söylediklerini metne dönüştürür, duygu ve konu analizini sunar.")

live_speaker_placeholder = st.empty()
final_results_placeholder = st.empty()

# Dinleme kuyruğu
data_queue = queue.Queue()
speaker_durations = defaultdict(int)  # Konuşma süreleri
speaker_texts = defaultdict(str)  # Konuşmacı metinleri
speaker_word_counts = defaultdict(int)  # Kelime sayıları

# Dinleme işlemi
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    data_queue.put(indata.copy())

# Kayıt işlemi
def start_recording():
    duration = 10  # Kayıt süresi (saniye)
    samplerate = 44100
    chunk_duration = 2  # Her segment 2 saniye
    blocksize = samplerate * chunk_duration
    
    # Ses kaydını başlat
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize, dtype='int16'):
        st.info("Kayıt devam ediyor, lütfen konuşun...")
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                indata = data_queue.get(timeout=0.1)
                indata = indata.astype(np.float32) / np.max(np.abs(indata))

                # Ses segmentini metne dönüştür
                audio_segment = (indata.flatten() * 32767).astype(np.int16).tobytes()
                try:
                    transcript = transcribe_audio(audio_segment, samplerate)
                except Exception:
                    transcript = "[Metne Dönüştürülemedi]"

                # Konuşmacıyı tanımla
                speaker_name = process_audio(indata, samplerate)
                speaker_texts[speaker_name] += f" {transcript}"
                speaker_word_counts[speaker_name] += len(transcript.split())
                speaker_durations[speaker_name] += chunk_duration

                # Duygu ve konu analizi
                emotion, score = analyze_emotion(transcript)
                topic = analyze_topic(transcript)

                # Ekranda göster
                live_speaker_placeholder.markdown(
                    f"### Konuşmacı: {speaker_name}, Kelime Sayısı: {speaker_word_counts[speaker_name]}\n"
                    f"Konuşma Metni ({speaker_name}): {speaker_texts[speaker_name]}\n"
                    f"Duygu: {emotion}, Skor: {score:.2f}%\n"
                    f"Konu: {topic}"
                )

            except queue.Empty:
                continue

    # Sonuçları özetle
    final_results_placeholder.markdown("## Sonuçlar")
    for speaker, total_time in speaker_durations.items():
        final_results_placeholder.markdown(
            f"**{speaker}: {total_time} saniye, {speaker_word_counts[speaker]} kelime**\n"
            f"**Konuşma Metni ({speaker}):** {speaker_texts[speaker]}"
        )

    # Konuşmacı sürelerini görselleştir
    fig, ax = plt.subplots()
    ax.pie(speaker_durations.values(), labels=speaker_durations.keys(), autopct="%1.1f%%", colors=["red", "blue"])
    ax.set_title("Konuşmacı Süreleri")
    st.pyplot(fig)

if st.button("Kaydı Başlat"):
    start_recording()










