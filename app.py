# %% [1] IMPORTS AND SETUP
import os
import whisper
import pyttsx3
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from transformers import MarianMTModel, MarianTokenizer

# Configure FFmpeg path
os.environ['PATH'] += os.pathsep + r'G:\ffmpeg\bin'


# %% [2] AUDIO RECORDING (OPTIMIZED)
def record_urdu_audio(filename="urdu_audio.wav", duration=7):
    try:
        fs = 44100  # Higher sample rate for Urdu
        print(f"\n🎤 Speak Urdu clearly for {duration} seconds...")

        # Record with device verification
        audio = sd.rec(int(duration * fs),
                       samplerate=fs,
                       channels=1,
                       dtype='float32')
        sd.wait()

        # Normalize and save
        audio = audio / np.max(np.abs(audio))
        write(filename, fs, audio)

        # Additional processing
        audio_segment = AudioSegment.from_wav(filename)
        audio_segment = audio_segment.normalize()
        audio_segment.export(filename, format="wav")

        print(f"✅ Saved to {os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return False


# %% [3] LOAD MODELS (WITH FALLBACKS)
def load_models():
    models = {}
    try:
        # Load Whisper (fallback to small if medium fails)
        try:
            models['whisper'] = whisper.load_model("medium")
        except:
            print("Falling back to whisper-small")
            models['whisper'] = whisper.load_model("small")

        # Translation model with Urdu focus
        model_name = "Helsinki-NLP/opus-mt-ur-en"
        models['tokenizer'] = MarianTokenizer.from_pretrained(model_name)
        models['translator'] = MarianMTModel.from_pretrained(model_name)

        # TTS setup
        models['tts'] = pyttsx3.init()
        models['tts'].setProperty('rate', 150)

        return models
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise


models = load_models()


# %% [4] IMPROVED TRANSCRIPTION
def transcribe_urdu(audio_path):
    try:
        # Urdu-specific parameters
        result = models['whisper'].transcribe(
            audio_path,
            language="ur",
            temperature=0.2,
            initial_prompt="یہ اردو میں گفتگو ہے",
            fp16=False
        )

        # Post-processing
        text = result["text"].strip()

        # Common Urdu corrections
        corrections = {
            "ہے": "ہے",
            "ہیں": "ہیں",
            "میں": "میں"
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)

        return text
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None


# %% [5] ENHANCED TRANSLATION
def translate_urdu(urdu_text):
    try:
        if not urdu_text or len(urdu_text) < 3:
            return None

        # Pre-processing
        text = urdu_text.replace("۔", ".").replace("ہے", " ہے ")

        # Translation
        inputs = models['tokenizer'](text, return_tensors="pt", truncation=True)
        outputs = models['translator'].generate(**inputs)
        english = models['tokenizer'].decode(outputs[0], skip_special_tokens=True)

        # Post-translation fixes
        fixes = {
            "I am": "I",
            "you are": "you",
            "he is": "he"
        }
        for wrong, correct in fixes.items():
            english = english.replace(wrong, correct)

        return english
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None


# %% [6] MAIN PIPELINE
def main():
    # 1. Record
    if not record_urdu_audio():
        return

    # 2. Transcribe
    print("\n🔍 Transcribing...")
    urdu_text = transcribe_urdu("urdu_audio.wav")
    if not urdu_text:
        return
    print(f"📝 Urdu: {urdu_text}")

    # 3. Translate
    print("\n🌍 Translating...")
    english_text = translate_urdu(urdu_text)
    if not english_text:
        return
    print(f"✅ English: {english_text}")

    # 4. Speak
    print("\n🔊 Speaking...")
    models['tts'].say(english_text)
    models['tts'].runAndWait()

    # Cleanup
    os.remove("urdu_audio.wav")


if __name__ == "__main__":
    main()