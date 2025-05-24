# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import subprocess
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load models once at startup
asr_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def convert_to_wav(input_path):
    output_path = tempfile.mktemp(suffix=".wav")
    subprocess.call([
        'ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', output_path,
        '-y', '-loglevel', 'error'
    ])
    return output_path

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            file.save(temp_file.name)
            input_path = temp_file.name

        # Convert to WAV
        wav_path = convert_to_wav(input_path)

        # Transcribe
        result = asr_model.transcribe(wav_path)
        transcript = result['text']

        # Summarize
        summary_result = summarizer(transcript, max_length=100, min_length=30, do_sample=False)
        summary = summary_result[0]['summary_text']

        return jsonify({
            "transcript": transcript,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(input_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # required by Render
    app.run(host="0.0.0.0", port=port, debug=False)

