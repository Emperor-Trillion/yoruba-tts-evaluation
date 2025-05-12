# Load model and dependencies
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import hf_hub_download
import torch
import soundfile as sf  # For saving audio files

# Load model and processor
processor = SpeechT5Processor.from_pretrained("imhotepai/yoruba-tts")
model = SpeechT5ForTextToSpeech.from_pretrained("imhotepai/yoruba-tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embedding
dir_ = hf_hub_download(repo_id="imhotepai/yoruba-tts", filename="speaker_embeddings.pt")
speaker_embeddings = torch.load(dir_)

# Yoruba speech list
speech_list = [
    "Kìí ṣe ọgbẹ́ni yẹni",
    "Ìgbẹ́kẹ̀lé Ọlọ́run dùn",
    "Àfẹ́ẹ̀ri",
    "Olúkòso",
    "Kí olórí kódi orí ara rẹ̀ mú",
    "Ìwàlẹwà ọmọ obìnrin"
]

# Generate and save each synthesized speech
for idx, text in enumerate(speech_list):
    text = text.lower()
    inputs = processor(text=text, return_tensors="pt")

    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save to file
    output_filename = f"yoruba_tts_output_{idx+1}.wav"
    sf.write(output_filename, speech.numpy(), samplerate=16000)
    print(f"Saved: {output_filename}")
