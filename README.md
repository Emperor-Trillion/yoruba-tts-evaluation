# yoruba-tts-evaluation
A lab-based evaluation of the Yorùbá Text-to-Speech model using SpeechT5

Title: Evaluating Yorùbá TTS Using SpeechT5: A Lab-Based Study on Model Performance and Challenges

Author: Sunday Emmanuel Sanni
Date: May 2025

---

 Introduction

Text-to-Speech (TTS) systems have rapidly advanced in the past decade, primarily for high-resource languages. However, languages like Yorùbá, spoken by over 40 million people, remain underrepresented in speech AI research. This short study investigates the quality and challenges of Yorùbá speech synthesis using a pre-trained TTS model built on Microsoft’s SpeechT5 architecture.

---

 Objective

The goal of this lab work was to evaluate the performance of the `imhotepai/yoruba-tts` model, a Yorùbá TTS system available on Hugging Face, through direct model interaction and objective quality metrics.

---

 Model Overview

- Base model: SpeechT5ForTextToSpeech
- Architecture: Transformer encoder-decoder
- Tokenizer: SentencePiece
- Input: Tokenized Yorùbá text
- Output: Mel-spectrograms and synthesized waveforms
- Conditioning: Speaker embeddings from pretrained file
- Vocoder: SpeechT5HifiGan

---

 Experimental Setup

- Platform: Google Colab
- Tools Used: Hugging Face Transformers, PyTorch, Librosa
- Model repo: [https://huggingface.co/ImhotepAI/yoruba-tts](https://huggingface.co/ImhotepAI/yoruba-tts)

Sample Input Text:

```python
speech_list = [
    "Ìfẹ́ ni ògbóni yànìyàn",
    "Àgbọ̀nkọ́lọ́ Olórun dáni",
    "Ìfẹ́ Ọlọ́run",
    "Olúkòso",
    "Olórí kòdi ọ̀rọ̀ ara rẹ̀ mú",
    "Ìwàlàwà ọmọ obìnrin"
]
```

- Sample rate: 16,000 Hz
- Voice identity: Default speaker\_embeddings.pt

---

 Evaluation Metrics

| Metric                                         | Score | Interpretation                         |
| ---------------------------------------------- | ----- | -------------------------------------- |
| MCD (Mel-Cepstral Distortion)                  | 77.46 | High spectral distortion, poor quality |
| PESQ (Perceptual Evaluation of Speech Quality) | 1.14  | Close to lowest possible score         |
| STOI (Short-Time Objective Intelligibility)    | 0.088 | Poor intelligibility                   |

---

 Findings

- The generated Yorùbá speech lacks natural prosody and phonetic precision.
- Severe degradation in clarity and tonal accuracy, which is critical for understanding in Yorùbá.
- Low objective scores indicate that pretraining alone is insufficient for usable Yorùbá TTS.

---

 Recommendations

- Fine-tune the model on clean, tonal Yorùbá datasets with diverse accents.
- Improve preprocessing: silence trimming, tone normalization, and better speaker embeddings.
- Explore multilingual pretraining or phoneme-based modeling.

---

 Conclusion

This experiment demonstrates the urgent need for robust, data-rich models for Yorùbá speech synthesis. It highlights the limitations of zero-shot TTS in tonal, low-resource languages and calls for more active contributions in dataset development and model fine-tuning.

> -This article is based on a lab assignment for the Artificial Intelligence course at Vilnius University – Šiauliai Academy, under Prof. Gintautas Daunys.-
