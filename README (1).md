# 🔤 Hindi Ambiguity Parser — Transformer (mT5-small)

> **NLP Assignment:** Discuss Ambiguity in NLP & Train a Transformer (LLM) to Handle Ambiguity in Hindi Language

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/hindi-ambiguity-parser/blob/main/Hindi_Ambiguity_Parser.ipynb)
![Model](https://img.shields.io/badge/Model-mT5--small-purple)
![Parameters](https://img.shields.io/badge/Parameters-~300M-blue)
![Language](https://img.shields.io/badge/Language-Hindi-orange)
![Framework](https://img.shields.io/badge/Framework-HuggingFace-yellow)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green)

---

## 📌 Project Overview

This project builds a **Hindi Ambiguity Parser** using a fine-tuned **mT5-small Transformer model**. Given an ambiguous Hindi sentence, the model detects that the sentence is ambiguous and generates **two different possible interpretations** of it.

```
Input:  "सोना अच्छा है।"
           ↓
    ┌──────────────────┐
    │  Ambiguity Found │
    └──────────────────┘
           ↓
  ┌────────┴────────┐
  ▼                 ▼
अर्थ 1              अर्थ 2
Gold is precious    Sleeping is good for health
```

---

## 🧠 Types of Ambiguity Covered

| Type | Hindi | Example | Both Meanings |
|------|-------|---------|---------------|
| **Lexical** | शाब्दिक | सोना अच्छा है | Gold is precious / Sleeping is healthy |
| **Syntactic** | वाक्यात्मक | मैंने उड़ते पक्षी को देखा | Bird was flying / I was flying |
| **Referential** | संदर्भात्मक | राम और श्याम मिले, वह खुश था | Ram was happy / Shyam was happy |
| **Pragmatic** | व्यावहारिक | यहाँ बहुत ठंड है | Stating a fact / Implied request |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Model** | `google/mt5-small` |
| **Architecture** | Multilingual T5 (Encoder-Decoder Transformer) |
| **Total Parameters** | ~556 Million (well under 2B limit ✅) |
| **Pre-training** | 101 languages including Hindi (mC4 corpus) |
| **Task** | Seq2Seq — Ambiguous sentence → Two interpretations |
| **Separator** | `[SEP]` token between interpretations |

---

## 📂 Repository Structure

```
hindi-ambiguity-parser/
│
├── Hindi_Ambiguity_Parser.ipynb   ← Main Google Colab notebook
├── README.md                      ← This file
├── writeup/
│   └── Project_Writeup.docx       ← Formal project report
└── assets/
    └── output_demo.png            ← Screenshot of output
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Click the **"Open in Colab"** badge above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**
4. Training takes ~50 minutes on GPU (10 epochs)

### Option 2: Run Locally
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hindi-ambiguity-parser.git
cd hindi-ambiguity-parser

# Install dependencies
pip install transformers datasets torch sentencepiece accelerate ipywidgets

# Open notebook
jupyter notebook Hindi_Ambiguity_Parser.ipynb
```

---

## 📦 Dependencies

```
transformers >= 4.30.0
datasets >= 2.0.0
torch >= 2.0.0
sentencepiece
accelerate
ipywidgets
pandas
```

Install all at once:
```bash
pip install transformers datasets torch sentencepiece accelerate ipywidgets pandas
```

---

## 📝 Dataset

- **25 hand-crafted ambiguous Hindi sentences**
- Each sentence annotated with:
  - `ambiguity_type` — category of ambiguity
  - `ambiguous_word` — the word/phrase causing ambiguity
  - `interpretation_1` — first possible meaning
  - `interpretation_2` — second possible meaning
- **80/20 train-test split** → 20 training, 5 testing

**Distribution:**
```
शाब्दिक  (Lexical)      → 14 sentences
वाक्यात्मक (Syntactic)   →  4 sentences
व्यावहारिक (Pragmatic)   →  4 sentences
अर्थपरक   (Semantic)     →  2 sentences
संदर्भात्मक (Referential) →  1 sentence
```

---

## 🏋️ Training Configuration

```python
Seq2SeqTrainingArguments(
    num_train_epochs            = 10,
    per_device_train_batch_size = 4,
    learning_rate               = 5e-4,
    warmup_steps                = 10,
    weight_decay                = 0.01,
    fp16                        = True,   # Half-precision on GPU
    predict_with_generate       = True,
)
```

**Training Results:**
- Final Loss: `0.0000`
- Training Time: `3244.9s` (~54 min on T4 GPU)

---

## 🔍 Inference

```python
result = analyze_ambiguity("सोना अच्छा है।")
# Returns:
# {
#   "sentence": "सोना अच्छा है।",
#   "is_ambiguous": True,
#   "interpretation_1": "सोना (स्वर्ण) बहुत कीमती है।",
#   "interpretation_2": "सोना (नींद लेना) स्वास्थ्य के लिए अच्छा है।"
# }
```

The notebook also includes an **interactive widget** — type any Hindi sentence and click **Analyze** to see both interpretations in a colourful HTML card.

---

## 🖼️ Output Preview

```
┌──────────────────────────────────────────────────────┐
│  NLP AMBIGUITY ANALYSIS · mT5-small                  │
│  🔍 Hindi Ambiguity Parser                           │
├──────────────────────────────────────────────────────┤
│  ⚠️  यह वाक्य अस्पष्ट (Ambiguous) है!              │
├──────────────────────────────────────────────────────┤
│  मूल वाक्य: सोना अच्छा है।                         │
├────────────────────┬─────────────────────────────────┤
│  📗 अर्थ 1         │  📘 अर्थ 2                      │
│  सोना (स्वर्ण)     │  सोना (नींद लेना)              │
│  बहुत कीमती है।    │  स्वास्थ्य के लिए अच्छा है।     │
└────────────────────┴─────────────────────────────────┘
```

---

## ⚠️ Known Issue & Fix

mT5 sometimes outputs **raw byte tokens** like `<0x03>` instead of Hindi text. The notebook includes a fix:

```python
import re
# Strip hex tokens
decoded = re.sub(r"<0x[0-9A-Fa-f]+>", "", decoded).strip()
# Count Hindi characters to detect garbage output
hindi_chars = re.findall(r'[\u0900-\u097F]', decoded)
if len(hindi_chars) < 5:
    # Fall back to training data directly
    match = next((d for d in RAW_DATA if d["hindi_sentence"] == sentence), None)
```

---

## 📖 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Transformer** | Neural network architecture based on Self-Attention mechanism |
| **mT5** | Multilingual Text-To-Text Transfer Transformer by Google |
| **Seq2Seq** | Maps input sequence to output sequence (used here for interpretation generation) |
| **Fine-tuning** | Adapting a pre-trained model to a specific task with small data |
| **Beam Search** | Decoding strategy that explores top-k sequences for better output |
| **[SEP] Token** | Custom separator to split two interpretations in model output |

---

## 👨‍💻 Author

**NLP Course Assignment**  
Topic: *Discuss Ambiguity in NLP. Train a transformer (LLM) to handle the ambiguity in Hindi language.*

---

## 📄 License

This project is for educational purposes only.
