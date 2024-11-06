# AI_TASK_TTS_STT
VITS Speech Synthesis Model for Non-Hindi Language


This project implements a speech synthesis model using the VITS (Variational Inference Text-to-Speech) architecture for a non-Hindi language dataset from the AI4Bharat corpus. The model generates high-quality audio from text, streaming WAV to FLAC via WebSockets.

Table of Contents
Project Overview
Setup
Data Preparation
Training the Model
Evaluation
Deployment
References
Project Overview
The goal of this project is to train a robust speech synthesis model using VITS on a non-Hindi language from the AI4Bharat data corpus. This model is trained to convert text to speech, delivering audio in real-time using WebSocket streaming.

Key features:

Non-Hindi language selection for uniqueness.
High-quality text-to-speech synthesis.
WAV to FLAC conversion over WebSockets.
Setup
Prerequisites
Python 3.8+
CUDA-enabled GPU for efficient training (recommended but optional)
PyTorch
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-vits-project.git
cd your-vits-project
Install required libraries:

bash
Copy code
pip install -r requirements.txt
If requirements.txt is unavailable, manually install key dependencies:

bash
Copy code
pip install torch torchaudio librosa soundfile numpy
Data Preparation
Dataset Acquisition:

Select a language from AI4Bharat’s Indic Voices or Svarah datasets.
Exclude any news datasets to comply with project requirements.
Organize Data: Place audio files, transcripts, and metadata in a data directory:

css
Copy code
data/
├── audio_files/
│   └── [audio files, e.g., .wav]
├── transcripts/
│   └── [text files, e.g., .txt]
└── meta/
    └── meta_speaker_stats.csv
Metadata File: Create a metadata.csv that maps audio files to transcripts, following the format:

Copy code
path_to_audio_file|transcript
Split Data: Use a script to divide metadata.csv into train_metadata.csv, val_metadata.csv, and test_metadata.csv.

Training the Model
Update Configuration: Modify config.json to specify paths for training and validation files:

json
Copy code
"data": {
  "training_files": "data/train_metadata.csv",
  "validation_files": "data/val_metadata.csv",
  "sample_rate": 22050,
  "n_mel_channels": 80,
  ...
}
Run Training: Start training the VITS model:

bash
Copy code
python train.py -c config.json
Training logs and checkpoints will be saved to the specified directory in config.json.

Evaluation
Evaluate Model: Run evaluation on the test set to assess model quality. Use visualization tools like TensorBoard to monitor spectrograms and loss functions.

Sample Audio Output: Generate audio samples from text inputs using infer.py or a custom script to validate quality.

Deployment
Setup WebSocket Server for WAV to FLAC Conversion: Use Flask or FastAPI to deploy the trained model. Example server code:

python
Copy code
from flask import Flask, request
import soundfile as sf
import io

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_wav_to_flac():
    wav_audio = request.files['file']
    wav_data, samplerate = sf.read(io.BytesIO(wav_audio.read()))
    flac_buffer = io.BytesIO()
    sf.write(flac_buffer, wav_data, samplerate, format='FLAC')
    flac_buffer.seek(0)
    return flac_buffer, 200, {'Content-Type': 'audio/flac'}

if __name__ == '__main__':
    app.run(port=5000)
Testing: Send sample WAV files to your endpoint and verify that it returns FLAC audio files.

References
VITS: VITS GitHub Repository
AI4Bharat Data Corpus: AI4Bharat
