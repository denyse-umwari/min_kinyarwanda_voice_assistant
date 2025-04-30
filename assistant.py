from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
from gtts import gTTS
import os
from difflib import get_close_matches
import os.path

# Initialize KinyaWhisper ASR
print("Loading KinyaWhisper model...")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
model = WhisperForConditionalGeneration.from_pretrained(
    "benax-rw/KinyaWhisper")

# Knowledge base
qa_pairs = {
    "Rwanda Coding Academy iherereye he?": "Iherereye mu Karere ka Nyabihu, mu Ntara y'Iburengerazuba.",
    "Umurwa mukuru w'u Rwanda ni uwuhe?": "Ni Kigali.",
    "U Rwanda ruri he?": "U Rwanda ruri mu Afurika y'Iburasirazuba.",
    "Amasaha yo gukora ni ayahe mu Rwanda?": "Kuva saa tatu za mu gitondo kugeza saa kumi n'imwe za nimugoroba.",
    "U Rwanda rufite abaturage bangahe?": "U Rwanda rufite abaturage barenga miliyoni 13."
}


def load_audio_file(filepath):
    """Load and resample audio file to 16kHz"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(filepath)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        return waveform.numpy(), sample_rate

    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise


def transcribe_audio(audio_array, sample_rate):
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def find_best_answer(question):
    matches = get_close_matches(question, qa_pairs.keys(), n=1, cutoff=0.6)
    return qa_pairs[matches[0]] if matches else "Ntabwo mbyumvise neza. Vugurure kandi wongere ubaze."


def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("./responses/response.mp3")
    os.system("start ./responses/response.mp3")


def save_transcription(text, filename="transcription_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcription saved to '{filename}'")


def process_audio_file(filepath):
    try:
        audio, sr = load_audio_file(filepath)
        question = transcribe_audio(audio, sr)
        print(f"🗣️ Transcription: {question}")
        save_transcription(question)

        answer = find_best_answer(question)
        print(f"🤖 Answer: {answer}")
        text_to_speech(answer)
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    audio_file = input("Enter path to Kinyarwanda audio file: ").strip('"')
    process_audio_file(audio_file)
