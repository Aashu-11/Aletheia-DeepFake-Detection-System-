import streamlit as st
import os
import shutil
import librosa
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import io
import tempfile
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from finalpredicted import predict_deepfake_enhanced
import pandas as pd
import json

def _load_audio(source):
    if isinstance(source, (bytes, bytearray)):
        return librosa.load(io.BytesIO(source), res_type="kaiser_fast")
    if hasattr(source, "getbuffer"):
        data = io.BytesIO(source.getbuffer())
        try:
            return librosa.load(data, res_type="kaiser_fast")
        except Exception:
            data.seek(0)
            try:
                audio, sample_rate = sf.read(data)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                return audio, sample_rate
            except Exception:
                data.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(data.read())
                    tmp_path = tmp.name
                try:
                    return librosa.load(tmp_path, res_type="kaiser_fast")
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
    return librosa.load(source, res_type="kaiser_fast")


def extract_features(file_path):
    try:
        audio, sample_rate = _load_audio(file_path)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        st.error(f"Error encountered while parsing audio file: {e}")
        return None

def _audio_confidence(model, features):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([features])[0]
        return float(max(prob))
    if hasattr(model, "decision_function"):
        score = model.decision_function([features])[0]
        return float(1 / (1 + np.exp(-score)))
    return None


def analyze_audio(example_file_path):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)

        loaded_model = joblib.load("svm_model.joblib")
        features = extract_features(example_file_path)
        if features is None:
            return None

        prediction = loaded_model.predict([features])[0]
        class_label = "Real" if prediction == 1 else "Fake"
        confidence = _audio_confidence(loaded_model, features)

        audio, sample_rate = _load_audio(example_file_path)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        rms = float(np.mean(librosa.feature.rms(y=audio)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))
        harmonic, percussive = librosa.effects.hpss(audio)
        hpr = float((np.sum(np.abs(harmonic)) + 1e-8) / (np.sum(np.abs(percussive)) + 1e-8))

        indicators = []
        if flatness > 0.2:
            indicators.append("High spectral flatness (noise-like energy).")
        if zcr > 0.1:
            indicators.append("Elevated zero-crossing rate (noisy or synthetic edges).")
        if spectral_centroid > 3500:
            indicators.append("High spectral centroid (brighter, artifact-heavy spectrum).")
        if hpr < 1.2:
            indicators.append("Low harmonic-to-percussive ratio (weaker vocal structure).")

        return {
            "label": class_label,
            "confidence": confidence,
            "mel_db": mel_db,
            "mfcc": mfcc,
            "sample_rate": sample_rate,
            "duration": float(len(audio) / sample_rate),
            "rms": rms,
            "spectral_centroid": spectral_centroid,
            "zcr": zcr,
            "flatness": flatness,
            "rolloff": rolloff,
            "hpr": hpr,
            "indicators": indicators,
            "audio": audio,
        }
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        return None
    
def check_video_enhanced(uploaded_video_file, method, use_tta=True):
    """Enhanced video checking with TTA and frequency/texture analysis"""
    with st.spinner("Analyzing video with enhanced detection..."):
        input_video_file_path = "uploaded_video.mp4"
        with open(input_video_file_path, "wb") as f:
            f.write(uploaded_video_file.getbuffer())
        
        # Use enhanced prediction with all features
        fake_prob, real_prob, pred, details = predict_deepfake_enhanced(
            input_video_file_path, 
            method, 
            return_details=True,
            use_tta=use_tta,
            verbose=False
        )
        
    return fake_prob, real_prob, pred, details


def render_mel_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Mel Spectrogram (dB)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Bands")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)


def render_waveform(audio, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 4))
    times = np.arange(len(audio)) / sample_rate
    ax.plot(times, audio, linewidth=0.8)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    plt.close(fig)


def render_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("MFCC")
    ax.set_xlabel("Time")
    ax.set_ylabel("Coefficient")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


def render_frame_gallery(frames_dir, key_prefix, captions=None):
    if not frames_dir or not os.path.isdir(frames_dir):
        st.info("No extracted frames found to display.")
        return

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        st.info("No extracted frames found to display.")
        return

    max_display = min(60, len(frame_paths))
    default_display = min(20, len(frame_paths))
    num_frames = st.slider(
        "Frames to display", 1, max_display, default_display, key=f"{key_prefix}_frames"
    )

    cols = st.columns(4)
    for i, frame_path in enumerate(frame_paths[:num_frames]):
        name = os.path.basename(frame_path)
        caption = name
        if captions and name in captions:
            caption = captions[name]
        cols[i % 4].image(frame_path, caption=caption, use_column_width=True)


def _select_frame_paths(frame_paths, max_frames):
    if len(frame_paths) <= max_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
    return [frame_paths[i] for i in indices]


def _fft_highpass(gray, center_frac=0.12):
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    rh, rw = int(h * center_frac), int(w * center_frac)
    magnitude[ch - rh:ch + rh, cw - rw:cw + rw] = 0
    band = max(1, int(min(h, w) * 0.01))
    magnitude[ch - band:ch + band, :] = 0
    magnitude[:, cw - band:cw + band] = 0
    return magnitude


def generate_gan_fft_images(frames_dir, out_dir, max_frames=40):
    os.makedirs(out_dir, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    frame_paths = _select_frame_paths(frame_paths, max_frames)

    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        magnitude = _fft_highpass(gray, center_frac=0.10)
        magnitude = np.log1p(magnitude)
        p99 = np.percentile(magnitude, 99)
        magnitude = magnitude / (p99 + 1e-8)
        magnitude = np.clip(magnitude, 0, 1)
        magnitude = np.power(magnitude, 0.5)
        thr = np.percentile(magnitude, 80)
        magnitude = np.where(magnitude >= thr, magnitude, 0)
        magnitude = np.uint8(255 * magnitude)
        spectrum = cv2.applyColorMap(magnitude, cv2.COLORMAP_INFERNO)
        out_path = os.path.join(out_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, spectrum)


def compute_fft_score(frames_dir, max_frames=30):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    frame_paths = _select_frame_paths(frame_paths, max_frames)
    if not frame_paths:
        return None

    scores = []
    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        magnitude = _fft_highpass(gray, center_frac=0.10)
        high_freq_energy = magnitude.sum()
        total_energy = np.abs(np.fft.fftshift(np.fft.fft2(gray))).sum() + 1e-8
        scores.append(high_freq_energy / total_energy)

    if not scores:
        return None
    return float(np.mean(scores))


def render_enhanced_analysis_metrics(details):
    """Render enhanced analysis metrics from the improved detection"""
    st.subheader("🔬 Enhanced Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Adaptive Threshold", 
            f"{details.get('adaptive_threshold', 0.5):.3f}",
            help="Dynamically adjusted threshold based on video characteristics"
        )
    
    with col2:
        st.metric(
            "Fake Fraction", 
            f"{details.get('fake_fraction', 0.5):.2f}",
            help="Percentage of frames needed to classify as fake"
        )
    
    with col3:
        uncertainty = details.get('avg_uncertainty', 0.0)
        st.metric(
            "Prediction Uncertainty", 
            f"{uncertainty:.4f}",
            delta=None if uncertainty < 0.05 else "High",
            delta_color="normal" if uncertainty < 0.05 else "inverse",
            help="Lower is better (< 0.05 is ideal)"
        )
    
    with col4:
        st.metric(
            "Frames Analyzed", 
            len(details.get('probabilities', [])),
            help="Total number of frames processed"
        )


def render_frequency_texture_analysis(details):
    """Render frequency domain and texture analysis results"""
    st.subheader("📊 Frequency & Texture Analysis")
    
    freq_analysis = details.get('freq_analysis', {})
    texture_analysis = details.get('texture_analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Frequency Domain Analysis**")
        
        freq_balance = freq_analysis.get('avg_freq_balance')
        high_freq = freq_analysis.get('avg_high_freq')
        
        if freq_balance is not None:
            # Visual indicator
            if freq_balance > 1.5:
                st.error(f"⚠️ High Frequency Imbalance: {freq_balance:.3f}")
                st.caption("Suspicious - Typical deepfakes have ratio > 1.5")
            elif freq_balance > 1.3:
                st.warning(f"⚡ Moderate Frequency Imbalance: {freq_balance:.3f}")
                st.caption("Borderline - May indicate manipulation")
            else:
                st.success(f"✓ Normal Frequency Balance: {freq_balance:.3f}")
                st.caption("Natural - Typical of real videos (1.0-1.3)")
            
            # Show bar
            st.progress(min(freq_balance / 2.0, 1.0))
        else:
            st.info("Frequency analysis not available")
        
        if high_freq is not None:
            st.metric("High Frequency Energy", f"{high_freq:.4f}")
    
    with col2:
        st.markdown("**Texture Pattern Analysis**")
        
        texture_uniformity = texture_analysis.get('avg_uniformity')
        texture_entropy = texture_analysis.get('avg_entropy')
        
        if texture_uniformity is not None:
            # Visual indicator
            if texture_uniformity < 0.08:
                st.error(f"⚠️ Low Texture Uniformity: {texture_uniformity:.3f}")
                st.caption("Suspicious - May indicate GAN artifacts")
            elif texture_uniformity < 0.12:
                st.warning(f"⚡ Moderate Texture Uniformity: {texture_uniformity:.3f}")
                st.caption("Borderline - Watch for other indicators")
            else:
                st.success(f"✓ Good Texture Uniformity: {texture_uniformity:.3f}")
                st.caption("Natural - Consistent skin texture")
            
            # Show bar
            st.progress(min(texture_uniformity * 10, 1.0))
        else:
            st.info("Texture analysis not available")
        
        if texture_entropy is not None:
            st.metric("Texture Entropy", f"{texture_entropy:.4f}")


def render_detection_confidence_breakdown(details, pred, fake_prob, real_prob):
    """Render detailed confidence breakdown"""
    st.subheader("🎯 Detection Confidence Breakdown")
    
    probabilities = details.get('probabilities', [])
    if not probabilities:
        st.info("No frame-level probabilities available")
        return
    
    probs_array = np.array(probabilities)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Probability Distribution**")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs_array, bins=30, alpha=0.7, color='coral' if pred == 1 else 'skyblue', edgecolor='black')
        ax.axvline(details.get('adaptive_threshold', 0.5), color='red', linestyle='--', linewidth=2, label='Adaptive Threshold')
        ax.set_xlabel('Fake Probability')
        ax.set_ylabel('Frame Count')
        ax.set_title('Distribution of Frame Predictions')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("**Confidence Statistics**")
        st.write({
            "Mean Probability": f"{np.mean(probs_array):.4f}",
            "Median Probability": f"{np.median(probs_array):.4f}",
            "Std Deviation": f"{np.std(probs_array):.4f}",
            "Min Probability": f"{np.min(probs_array):.4f}",
            "Max Probability": f"{np.max(probs_array):.4f}",
        })
        
        # Classification breakdown
        threshold = details.get('adaptive_threshold', 0.5)
        fake_frames = np.sum(probs_array >= threshold)
        real_frames = np.sum(probs_array < threshold)
        
        st.markdown("**Frame Classification**")
        st.write({
            "Fake Frames": f"{fake_frames} ({fake_frames/len(probs_array)*100:.1f}%)",
            "Real Frames": f"{real_frames} ({real_frames/len(probs_array)*100:.1f}%)",
            "Total Frames": len(probs_array)
        })


def render_probability_timeline(probabilities):
    """Render frame-by-frame probability timeline"""
    if not probabilities:
        return
    
    st.subheader("📈 Frame-by-Frame Analysis Timeline")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    frames = list(range(len(probabilities)))
    ax.plot(frames, probabilities, marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Base Threshold (0.5)')
    ax.fill_between(frames, probabilities, 0.5, where=(np.array(probabilities) >= 0.5), 
                     alpha=0.3, color='red', label='Classified as Fake')
    ax.fill_between(frames, probabilities, 0.5, where=(np.array(probabilities) < 0.5), 
                     alpha=0.3, color='green', label='Classified as Real')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Fake Probability')
    ax.set_title('Temporal Analysis of Deepfake Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Enhanced Deepfake Checker", page_icon="🔍", layout="wide")
    
    st.title("🔍 Enhanced Deepfake Detection System")
    st.markdown("""
    This advanced system uses **multi-modal analysis** including:
    - 🧠 Deep learning model predictions with Test-Time Augmentation
    - 🌈 Enhanced Grad-CAM heatmap visualization (multi-layer)
    - 📊 Frequency domain analysis (FFT patterns)
    - 🧵 Texture pattern analysis (Local Binary Patterns)
    - 🎯 Adaptive thresholding based on video characteristics
    """)
    
    st.header("🎵 Audio Deepfake Detection")
    uploaded_audio_file = st.file_uploader("Upload Audio File", type=["wav"], key="audio_uploader")
    if uploaded_audio_file is not None:
        st.write("Uploaded audio file details:")
        audio_file_details = {"FileName": uploaded_audio_file.name, "FileType": uploaded_audio_file.type, "FileSize": uploaded_audio_file.size}
        st.write(audio_file_details)

        if st.button("Check Audio"):
            with st.spinner("Checking audio..."):
                audio_report = analyze_audio(uploaded_audio_file)
            if audio_report is None:
                st.error("Error extracting audio features.")
            else:
                confidence = audio_report.get("confidence")
                conf_text = "N/A" if confidence is None else f"{round(confidence * 100, 2)}%"
                st.subheader("Audio Report")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", audio_report["label"].upper())
                c2.metric("Confidence (%)", conf_text)
                c3.metric("Duration (s)", f"{round(audio_report['duration'], 2)}")

                st.write("Signal metrics")
                st.write(
                    {
                        "sample_rate": audio_report["sample_rate"],
                        "rms": round(audio_report["rms"], 6),
                        "spectral_centroid": round(audio_report["spectral_centroid"], 2),
                        "zero_crossing_rate": round(audio_report["zcr"], 6),
                        "spectral_flatness": round(audio_report["flatness"], 6),
                        "spectral_rolloff": round(audio_report["rolloff"], 2),
                        "harmonic_percussive_ratio": round(audio_report["hpr"], 4),
                    }
                )
                render_waveform(audio_report["audio"], audio_report["sample_rate"])
                render_mel_spectrogram(audio_report["mel_db"])
                render_mfcc(audio_report["mfcc"])

                st.subheader("Audio Explainability")
                if audio_report["indicators"]:
                    st.write(audio_report["indicators"])
                else:
                    st.write("No strong artifact indicators detected by heuristics.")
            
    st.markdown("---")
    st.header("🎬 Video Deepfake Detection")
    
    # Settings
    with st.expander("⚙️ Detection Settings"):
        use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True, 
                             help="Runs multiple augmented versions for more robust predictions")
        st.info("TTA improves accuracy but takes ~30% longer to process")
    
    uploaded_video_file = st.file_uploader("Choose a video file", type=["mp4"], key="video_uploader")
    method_mapping = {"MTCNN (Plain Frames)": "plain_frames"}

    if uploaded_video_file is not None:
        selected_option = st.selectbox("Select detection method", list(method_mapping.keys()))
        st.video(uploaded_video_file)

        method = method_mapping[selected_option]

        if st.button("🚀 Analyze Video with Enhanced Detection"):
            fake_prob, real_prob, pred, details = check_video_enhanced(uploaded_video_file, method, use_tta)

            if pred is None:
                st.error("❌ Failed to detect DeepFakes in the video.")
            else:
                label = "REAL" if pred == 0 else "DEEPFAKE"
                probability = real_prob if pred == 0 else fake_prob
                if probability is not None:
                    probability = round(probability * 100, 4)
                total_frames = len(details.get("probabilities", [])) if details else 0

                # Main result banner
                st.markdown("---")
                if label == "REAL":
                    st.success(f"✅ The video is classified as **{label}** with {probability}% confidence")
                else:
                    st.error(f"⚠️ The video is classified as **{label}** with {probability}% confidence")

                if details:
                    # Enhanced metrics section
                    render_enhanced_analysis_metrics(details)
                    
                    st.markdown("---")
                    
                    # Frequency and texture analysis
                    render_frequency_texture_analysis(details)
                    
                    st.markdown("---")
                    
                    # Confidence breakdown
                    render_detection_confidence_breakdown(details, pred, fake_prob, real_prob)
                    
                    st.markdown("---")
                    
                    # Timeline
                    probabilities = details.get("probabilities", [])
                    if probabilities:
                        render_probability_timeline(probabilities)
                    
                    st.markdown("---")
                    
                    # Detailed tabs
                    st.subheader("📑 Detailed Analysis Tabs")
                    tabs = st.tabs([
                        "Enhanced Heatmaps",
                        "Frequency Spectrum (FFT)",
                        "Frame Gallery",
                        "Technical Details",
                        "Data Export"
                    ])

                    frames_dir = details.get("frames_dir")
                    grad_cam_dir = details.get("grad_cam_dir")
                    video_root = os.path.dirname(os.path.dirname(frames_dir)) if frames_dir else ""
                    gan_dir = os.path.join(video_root, "gan_fft") if video_root else ""

                    with tabs[0]:
                        st.markdown("### 🌡️ Enhanced Multi-Layer Grad-CAM Heatmaps")
                        st.info("These heatmaps show which regions the model focuses on. Red areas indicate suspicious regions.")
                        
                        if grad_cam_dir and os.path.isdir(grad_cam_dir):
                            captions = {}
                            for item in details.get("grad_cam_frames", []):
                                prob_pct = round(item.get("prob", 0.0) * 100, 2)
                                captions[item.get("file", "")] = f"{item.get('file', '')} | Fake: {prob_pct}%"
                            render_frame_gallery(grad_cam_dir, "enhanced_grad_cam", captions=captions)
                            
                            st.markdown("**Heatmap Improvements:**")
                            st.write("✓ Uses last 3 convolutional layers (vs 1 in old version)")
                            st.write("✓ 15 smoothing samples (vs 8)")
                            st.write("✓ Adaptive normalization with gamma correction")
                            st.write("✓ Better color mapping for visualization")
                        else:
                            st.warning("Enhanced Grad-CAM outputs not found.")

                    with tabs[1]:
                        st.markdown("### 📊 Frequency Domain Analysis (FFT Spectrum)")
                        st.info("Deepfakes often have abnormal frequency patterns due to GAN artifacts.")
                        
                        if frames_dir and os.path.isdir(frames_dir):
                            if os.path.isdir(gan_dir):
                                shutil.rmtree(gan_dir)
                            generate_gan_fft_images(frames_dir, gan_dir)
                            
                            fft_score = compute_fft_score(frames_dir)
                            if fft_score:
                                st.metric("High-Frequency Score", f"{fft_score:.4f}")
                                if fft_score > 0.15:
                                    st.warning("High frequency score suggests possible manipulation")
                                else:
                                    st.success("Frequency patterns appear normal")
                            
                            render_frame_gallery(gan_dir, "freq_spectrum")
                        else:
                            st.warning("Frames directory not found for frequency analysis.")

                    with tabs[2]:
                        st.markdown("### 🖼️ Extracted Frame Gallery")
                        if frames_dir and os.path.isdir(frames_dir):
                            render_frame_gallery(frames_dir, "original_frames")
                        else:
                            st.info("No frames available.")

                    with tabs[3]:
                        st.markdown("### 🔧 Technical Analysis Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Model Configuration**")
                            st.json({
                                "architecture": "tf_efficientnet_b0_ns",
                                "input_size": "224x224",
                                "method": method,
                                "tta_enabled": use_tta,
                                "grad_cam_layers": 3,
                                "smoothing_samples": 15
                            })
                        
                        with col2:
                            st.markdown("**Analysis Parameters**")
                            st.json({
                                "adaptive_threshold": details.get('adaptive_threshold'),
                                "fake_fraction": details.get('fake_fraction'),
                                "avg_uncertainty": details.get('avg_uncertainty'),
                                "frames_analyzed": total_frames
                            })
                        
                        if probabilities:
                            st.markdown("**Probability Chart**")
                            st.line_chart(probabilities)

                    with tabs[4]:
                        st.markdown("### 💾 Export Analysis Data")
                        
                        # Prepare export data
                        export_data = {
                            "video_file": uploaded_video_file.name,
                            "prediction": label,
                            "confidence_percent": probability,
                            "analysis_details": {
                                "adaptive_threshold": details.get('adaptive_threshold'),
                                "fake_fraction": details.get('fake_fraction'),
                                "avg_uncertainty": details.get('avg_uncertainty'),
                                "frames_analyzed": total_frames
                            },
                            "frequency_analysis": details.get('freq_analysis', {}),
                            "texture_analysis": details.get('texture_analysis', {}),
                            "frame_probabilities": probabilities
                        }
                        
                        # JSON export
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="📥 Download Analysis (JSON)",
                            data=json_str,
                            file_name=f"deepfake_analysis_{uploaded_video_file.name}.json",mime="application/json")
                        # CSV export for frame data
                    if probabilities and details.get("filenames"):
                        df_frames = pd.DataFrame({
                            "frame": details.get("filenames"),
                            "fake_probability": probabilities,
                            "classification": ["FAKE" if p >= details.get('adaptive_threshold', 0.5) else "REAL" for p in probabilities]
                        })
                        csv = df_frames.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Frame Analysis (CSV)",
                            data=csv,
                            file_name=f"frame_analysis_{uploaded_video_file.name}.csv",
                            mime="text/csv"
                        )

            st.markdown("---")
            if st.button("🧹 Clean up output files"):
                if os.path.isdir("output"):
                    shutil.rmtree("output")
                    st.success("✅ Output files removed.")
                if os.path.isfile("uploaded_video.mp4"):
                    os.remove("uploaded_video.mp4")
                    st.success("✅ Uploaded video removed.")


if __name__ == "__main__":
    main()
 
