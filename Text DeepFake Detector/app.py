import streamlit as st
import torch
import torch.nn.functional as F
import nltk
import string
import math

from transformers.models.gpt2 import GPT2Tokenizer, GPT2LMHeadModel
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from collections import Counter
import plotly.express as px

# ==================== SETUP ====================
nltk.download("punkt")
nltk.download("stopwords")

device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()

stop_words = set(stopwords.words("english"))

# ==================== METRIC FUNCTIONS ====================

def perplexity_score(text):
    if not text.strip():
        return float("inf")
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def burstiness_score(text):
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    freq = FreqDist(tokens)
    repeated = sum(1 for c in freq.values() if c > 1)
    return repeated / len(freq)


def lexical_diversity(text):
    tokens = nltk.word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    return sum(len(nltk.word_tokenize(s)) for s in sentences) / len(sentences)


def entropy_score(text):
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = sum(freq.values())
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def stopword_ratio(text):
    tokens = nltk.word_tokenize(text.lower())
    return sum(1 for t in tokens if t in stop_words) / len(tokens) if tokens else 0.0


def repetition_ratio(text):
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    repeated = sum(c for c in freq.values() if c > 1)
    return repeated / len(tokens)


def unique_bigram_ratio(text):
    tokens = nltk.word_tokenize(text.lower())
    if len(tokens) < 2:
        return 0.0
    bigrams = list(ngrams(tokens, 2))
    return len(set(bigrams)) / len(bigrams)


def punctuation_density(text):
    return sum(1 for c in text if c in string.punctuation) / len(text) if text else 0.0


def content_word_ratio(text):
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    content = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return len(content) / len(tokens)

# ==================== VISUAL FUNCTIONS ====================

def plot_top_words(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    counts = Counter(tokens).most_common(10)
    if not counts:
        st.info("No significant words found.")
        return
    words, values = zip(*counts)
    fig = px.bar(x=words, y=values, title="Top Repeated Content Words")
    st.plotly_chart(fig, use_container_width=True)


def plot_sentence_length_distribution(text):
    sentences = nltk.sent_tokenize(text)
    lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    if lengths:
        fig = px.histogram(x=lengths, nbins=10, title="Sentence Length Distribution")
        st.plotly_chart(fig, use_container_width=True)


def plot_sentence_entropy(text):
    sentences = nltk.sent_tokenize(text)
    entropies = []
    for s in sentences:
        tokens = nltk.word_tokenize(s.lower())
        if not tokens:
            entropies.append(0)
            continue
        freq = Counter(tokens)
        total = sum(freq.values())
        entropies.append(-sum((c / total) * math.log2(c / total) for c in freq.values()))
    if entropies:
        fig = px.line(y=entropies, title="Sentence-Level Entropy Variation")
        st.plotly_chart(fig, use_container_width=True)


def plot_repetition_heatmap(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    freq = Counter(tokens).most_common(15)
    if freq:
        words, counts = zip(*freq)
        fig = px.imshow([counts], x=words, color_continuous_scale="Reds",
                        title="Repetition Intensity Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# ==================== UI ====================

st.set_page_config(page_title="Deepfake Text Detector", layout="wide")
st.title("🧠 Deepfake Text Detection Dashboard")
st.caption("Multi-metric, explainable analysis for AI-generated text")

text = st.text_area("Paste the text to analyze:", height=220)

if st.button("Analyze Text") and text.strip():

    with st.spinner("Analyzing text..."):
        ppl = perplexity_score(text)
        burst = burstiness_score(text)
        lex = lexical_diversity(text)
        sent_len = avg_sentence_length(text)
        entropy = entropy_score(text)
        stop_ratio = stopword_ratio(text)
        rep_ratio = repetition_ratio(text)
        bigram_ratio = unique_bigram_ratio(text)
        punct_density = punctuation_density(text)
        content_ratio = content_word_ratio(text)

    st.divider()
    st.subheader("📊 Linguistic Metrics")

    r1, r2, r3 = st.columns(3)
    r1.metric("Perplexity", f"{ppl:.2f}")
    r2.metric("Burstiness", f"{burst:.2f}")
    r3.metric("Lexical Diversity", f"{lex:.2f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("Avg Sentence Length", f"{sent_len:.1f}")
    r5.metric("Entropy", f"{entropy:.2f}")
    r6.metric("Stopword Ratio", f"{stop_ratio:.2f}")

    r7, r8, r9, r10 = st.columns(4)
    r7.metric("Repetition Ratio", f"{rep_ratio:.2f}")
    r8.metric("Unique Bigram Ratio", f"{bigram_ratio:.2f}")
    r9.metric("Punctuation Density", f"{punct_density:.3f}")
    r10.metric("Content Word Ratio", f"{content_ratio:.2f}")

    st.divider()
    st.subheader("🧾 Overall Assessment")

    if ppl > 40000 or burst < 0.24:
        st.error("⚠️ Likely AI-Generated Text")
    else:
        st.success("✅ Likely Human-Written Text")

    st.info("This tool provides probabilistic signals, not definitive proof.")

    st.divider()
    st.subheader("📈 Visual Analysis")

    c1, c2 = st.columns(2)
    with c1:
        plot_sentence_length_distribution(text)
        plot_repetition_heatmap(text)
    with c2:
        plot_sentence_entropy(text)

    st.divider()
    st.subheader("📊 Content Word Analysis")
    plot_top_words(text)
