"""
Streamlit Web Interface for Multi-Modal Fake News & Deepfake Detection System
"""

import sys
import os
import streamlit as st

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modalities.text_classifier import TextFakeNewsClassifier
from src.modalities.fact_checker import FactCheckerRAG
from src.modalities.audio_classifier import AudioFakeClassifier
from src.modalities.video_classifier import VideoDeepfakeClassifier

st.set_page_config(
    page_title="Multi-Modal Fake News & Deepfake Detector",
    page_icon="🛡️",
    layout="wide"
)

# Inject Custom CSS for Exciting High-Tech Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 35%, #312e81 65%, #4338ca 100%);
        padding: 2.2rem 2.2rem;
        border-radius: 20px;
        box-shadow: 0 10px 35px -8px rgba(67, 56, 202, 0.45);
        border: 1px solid rgba(165, 180, 252, 0.25);
        margin-bottom: 2rem;
        color: #ffffff;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 50%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        font-weight: 400;
        margin-bottom: 0.8rem;
    }

    .pill-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(10px);
        color: #f8fafc;
        margin-right: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Primary Gradient Button */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #d946ef 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.65rem 1.8rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 22px rgba(217, 70, 239, 0.6) !important;
    }

    /* Sidebar Header Accent */
    [data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
</style>

<div class="hero-banner">
    <div class="hero-title">🛡️ Multi-Modal Fake News & Deepfake Detector</div>
    <div class="hero-subtitle">Next-Generation AI Verification across Text Claims, Video Deepfakes, and Synthetic Voice Clones</div>
    <div>
        <span class="pill-badge">📰 Text Verification</span>
        <span class="pill-badge">🎥 Video Deepfake AI</span>
        <span class="pill-badge">🎙️ Voice Authenticity</span>
        <span class="pill-badge">🔍 Live Fact RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.header("⚡ Modality Control Panel")
modality = st.sidebar.radio("Select Analysis Engine", ["Text Claim", "Video Clip", "Audio Recording"])


# Text Modality Tab
if modality == "Text Claim":
    st.subheader("📰 Text Claim & Fact Verification")
    user_text = st.text_area(
        "Enter news headline, claim, or article snippet to analyze:",
        height=150,
        placeholder="e.g., Satya Niketan Collapse: DU Students Demand Accountability..."
    )
    
    if st.button("Analyze Text", type="primary"):
        if user_text:
            with st.spinner("Classifying text and searching live fact-check databases..."):
                classifier = TextFakeNewsClassifier()
                fact_checker = FactCheckerRAG()

                ml_result = classifier.predict(user_text)
                fact_result = fact_checker.verify_claim(user_text)

                if fact_result.get("matched"):
                    if fact_result.get("is_real", True):
                        final_label = "REAL NEWS"
                        final_confidence = 96.5
                        verdict_type = "success"
                        verdict_note = f"Confirmed against published news: **{fact_result.get('publisher', 'Verified Source')}**"
                    else:
                        final_label = "FAKE / DEBUNKED NEWS"
                        final_confidence = 94.0
                        verdict_type = "error"
                        verdict_note = f"Debunked by Fact-Check Source: **{fact_result.get('publisher', 'Verified Source')}**"
                else:
                    final_label = ml_result["label"].upper()
                    final_confidence = ml_result["confidence"]
                    verdict_type = "error" if "FAKE" in final_label else "success"
                    verdict_note = f"Analysis Engine: {ml_result['model_used']}"

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Integrated Verdict")
                    if verdict_type == "success":
                        st.success(f"### VERDICT: {final_label}")
                    else:
                        st.error(f"### VERDICT: {final_label}")
                    
                    st.progress(final_confidence / 100)
                    st.write(f"**Overall Confidence Score:** {final_confidence}%")
                    st.caption(verdict_note)

                with col2:
                    st.subheader("🔍 Live Fact Verification")
                    if fact_result.get("matched"):
                        if fact_result.get("is_real", True):
                            st.info(f"🛡️ **Verified Article Found (Trusted Publisher)!**")
                        else:
                            st.error(f"⚠️ **Debunked Claim Found!**")

                        st.markdown(f"**Headline:** [{fact_result['title']}]({fact_result['url']})")
                        st.markdown(f"**Publisher:** `{fact_result.get('publisher', 'News Source')}` ✅ *(Trusted Source)*")
                        st.markdown(f"**Status:** `{fact_result.get('rating', 'Verified News Event')}`")
                    else:
                        st.warning("No verified article found from trusted news publishers. Unverified blogs/sites are automatically filtered out.")

        else:
            st.warning("Please enter some text to analyze.")

# Video Modality Tab
elif modality == "Video Clip":
    st.subheader("🎥 Video Deepfake & Automatic Spoken News Verification")
    st.caption("Automatically detects visual face manipulation and transcribes spoken audio to verify factual news accuracy.")
    
    uploaded_video = st.file_uploader("Upload a video clip (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("Analyze Video & Fact-Check News", type="primary"):
            with st.spinner("Analyzing visual face frames & automatically transcribing audio track..."):
                temp_video_path = os.path.join("data", "temp_video." + uploaded_video.name.split(".")[-1])
                os.makedirs("data", exist_ok=True)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                video_classifier = VideoDeepfakeClassifier()
                video_result = video_classifier.predict(temp_video_path)

                audio_classifier = AudioFakeClassifier()
                asr_result = audio_classifier.transcribe_audio_automatically(temp_video_path)
                extracted_transcript = asr_result.get("transcript", "").strip()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎥 1. Visual Deepfake Analysis")
                    is_visual_fake = "Deepfake" in video_result["label"] or "Manipulated" in video_result["label"]
                    if is_visual_fake:
                        st.error(f"### Visuals: {video_result['label']}")
                    else:
                        st.success(f"### Visuals: {video_result['label']}")
                    st.progress(video_result["confidence"] / 100)
                    st.write(f"**Visual Confidence Score:** {video_result['confidence']}%")
                    st.caption(f"Analysis Engine: {video_result['model_used']}")

                with col2:
                    st.subheader("📰 2. Factual News Accuracy")
                    if extracted_transcript:
                        st.markdown(f"🗣️ **Auto-Extracted Video Speech:** *\"{extracted_transcript[:150]}...\"*")
                        
                        fact_checker = FactCheckerRAG()
                        text_classifier = TextFakeNewsClassifier()
                        
                        fact_result = fact_checker.verify_claim(extracted_transcript)
                        
                        if fact_result.get("matched"):
                            if fact_result.get("is_real", True):
                                st.success("### Factual News: REAL NEWS")
                                st.markdown(f"**Verified Headline:** [{fact_result['title']}]({fact_result['url']})")
                                st.markdown(f"**Publisher:** `{fact_result.get('publisher', 'Trusted Source')}` ✅ *(Trusted Source)*")
                            else:
                                st.error("### Factual News: FAKE / DEBUNKED NEWS")
                                st.markdown(f"**Fact Check:** [{fact_result['title']}]({fact_result['url']})")
                                st.markdown(f"**Rating:** `{fact_result.get('rating', 'Debunked Claim')}`")
                        else:
                            # Fallback to Text ML Classifier if live search has no exact match
                            ml_res = text_classifier.predict(extracted_transcript)
                            if "FAKE" in ml_res["label"].upper():
                                st.error(f"### Factual News: {ml_res['label'].upper()}")
                            else:
                                st.success(f"### Factual News: {ml_res['label'].upper()}")
                            st.progress(ml_res["confidence"] / 100)
                            st.write(f"**Classification Confidence:** {ml_res['confidence']}%")
                            st.caption(f"Analysis Engine: {ml_res['model_used']}")
                    else:
                        st.warning("⚠️ No spoken speech could be automatically transcribed from this video clip (video may be silent or audio non-English). Ensure the video has clear English speech.")

                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)


# Audio Modality Tab
elif modality == "Audio Recording":
    st.subheader("🎙️ Automatic Audio Voice & News Verification")
    st.caption("Analyzes voice authenticity (Human vs. AI Synthetic Voice) and spoken news factual accuracy.")
    
    uploaded_audio = st.file_uploader("Upload an audio clip (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("Analyze Audio & Fact-Check Spoken Claim", type="primary"):
            with st.spinner("Analyzing voice patterns, transcribing spoken audio, and checking live news..."):
                temp_path = os.path.join("data", "temp_audio." + uploaded_audio.name.split(".")[-1])
                os.makedirs("data", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                audio_classifier = AudioFakeClassifier()
                voice_result = audio_classifier.predict_voice_authenticity(temp_path)

                asr_result = audio_classifier.transcribe_audio_automatically(temp_path)
                transcript_text = asr_result.get("transcript", "").strip()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎙️ 1. Voice Authenticity")
                    if voice_result.get("is_synthetic"):
                        st.error(f"### Voice: {voice_result['voice_label']}")
                    else:
                        st.success(f"### Voice: {voice_result['voice_label']}")
                    st.progress(voice_result["voice_confidence"] / 100)
                    st.write(f"**Voice Confidence:** {voice_result['voice_confidence']}%")
                    st.caption(f"Analysis Engine: {voice_result['model_used']}")

                with col2:
                    st.subheader("📰 2. Factual News Accuracy")
                    if transcript_text:
                        st.markdown(f"🗣️ **Spoken Transcript:** *\"{transcript_text[:150]}...\"*")
                        fact_checker = FactCheckerRAG()
                        text_classifier = TextFakeNewsClassifier()
                        
                        fact_result = fact_checker.verify_claim(transcript_text)
                        
                        if fact_result.get("matched"):
                            if fact_result.get("is_real", True):
                                st.success("### Factual News: REAL NEWS")
                                st.markdown(f"**Verified Headline:** [{fact_result['title']}]({fact_result['url']})")
                                st.markdown(f"**Publisher:** `{fact_result.get('publisher', 'Trusted Source')}` ✅ *(Trusted Source)*")
                            else:
                                st.error("### Factual News: FAKE / DEBUNKED NEWS")
                                st.markdown(f"**Fact Check:** [{fact_result['title']}]({fact_result['url']})")
                                st.markdown(f"**Rating:** `{fact_result.get('rating', 'Debunked Claim')}`")
                        else:
                            # Fallback to Text ML Classifier if search has no match
                            ml_res = text_classifier.predict(transcript_text)
                            if "FAKE" in ml_res["label"].upper():
                                st.error(f"### Factual News: {ml_res['label'].upper()}")
                            else:
                                st.success(f"### Factual News: {ml_res['label'].upper()}")
                            st.progress(ml_res["confidence"] / 100)
                            st.write(f"**Classification Confidence:** {ml_res['confidence']}%")
                            st.caption(f"Analysis Engine: {ml_res['model_used']}")
                    else:
                        st.info("ℹ️ Ensure audio contains clear spoken English to automatically transcribe and verify news factual accuracy.")

                if os.path.exists(temp_path):
                    os.remove(temp_path)


