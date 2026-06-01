# Speech-to-Text, Sentiment Analysis, and Text-to-Speech Workflow

This project demonstrates an end-to-end workflow for processing audio input, performing speech-to-text conversion, analyzing the sentiment of the transcribed text, and generating an audio response based on that sentiment using various Hugging Face `transformers` models.

## Architecture Overview

The workflow follows a clear sequence of operations:

`Audio Input -> Speech-to-Text (STT) -> Text Transcription -> Tokenization -> Sentiment Analysis -> Sentiment-based Text Response -> Text-to-Speech (TTS) -> Audio Output`

Each step leverages powerful pre-trained models to achieve its specific task, enabling a comprehensive audio processing and intelligent response system.

## Tech Stack and Models

The core of this project relies on the following Python libraries and Hugging Face `transformers` models:

*   **Python Libraries:**
    *   `transformers`: For easily utilizing pre-trained machine learning models.
    *   `soundfile`: For reading and writing audio files.
    *   `librosa`: For advanced audio processing, specifically resampling.
    *   `numpy`: For numerical operations on audio data.
    *   `torch`: The underlying deep learning framework.

*   **Hugging Face Models:**
    *   **Speech-to-Text (STT): `openai/whisper-small`**
        *   **Description:** A robust Automatic Speech Recognition (ASR) model developed by OpenAI, capable of transcribing spoken language into text. The `small` version is chosen for its balance between performance and computational efficiency.
        *   **Role in Workflow:** Converts the uploaded audio file into its corresponding textual representation.

    *   **Sentiment Analysis: `distilbert-base-uncased-finetuned-sst-2-english`**
        *   **Description:** A distilled version of the BERT model, fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset for sentiment classification. It efficiently determines whether a given text expresses a positive or negative sentiment.
        *   **Role in Workflow:** Analyzes the transcribed text to identify the overall sentiment (e.g., positive, negative, neutral).

    *   **Text-to-Speech (TTS): `suno/bark-small`**
        *   **Description:** A transformer-based text-to-speech model that generates natural-sounding speech from text. The `small` version is used for quicker generation times.
        *   **Role in Workflow:** Synthesizes an audio response based on the sentiment detected in the original audio input.

## Workflow Steps (Detailed)

1.  **Environment Setup:** Installs all necessary libraries (`transformers`, `soundfile`, `datasets`, `accelerate`, `torch`, `sentencepiece`, `optimum`, `torchaudio`).
2.  **Model Loading:** Initializes the `pipeline` for STT, Sentiment Analysis, and TTS using the specified Hugging Face models.
3.  **Audio Upload:** Provides a mechanism for the user to upload an audio file (e.g., `.wav`).
4.  **Audio Preprocessing:** Loads the uploaded audio, and if necessary, resamples it to 16kHz, which is the optimal sampling rate for the Whisper model.
5.  **Speech-to-Text Transcription:** The preprocessed audio is fed into the `openai/whisper-small` pipeline to generate a text transcription.
6.  **Tokenization (Demonstrative):** The transcribed text is explicitly tokenized using the tokenizer associated with the sentiment analysis model to show the internal token representation. (Note: The sentiment pipeline handles this internally during its execution).
7.  **Sentiment Analysis:** The transcribed text is passed to the `distilbert-base-uncased-finetuned-sst-2-english` pipeline to determine its sentiment (e.g., 'POSITIVE', 'NEGATIVE').
8.  **Response Generation:** Based on the identified sentiment, a suitable text response is dynamically crafted. For instance, a negative sentiment might trigger an apologetic message, while a positive sentiment could generate a thank you message.
9.  **Text-to-Speech Generation:** The generated text response is converted into an audio file using the `suno/bark-small` pipeline.
10. **Audio Output:** The synthesized audio response is saved as a `.wav` file, providing an audio-based interaction.

## Usage

To use this workflow:

1.  **Run all cells sequentially** in the Colab notebook.
2.  When prompted, **upload an audio file** (e.g., `uploaded_audio.wav`).
3.  The notebook will then **process the audio**, **transcribe the speech**, **analyze its sentiment**, **generate a text response**, and finally **create an audio file** with the synthesized response.
4.  The generated `sentiment_response.wav` file will be available for download.
