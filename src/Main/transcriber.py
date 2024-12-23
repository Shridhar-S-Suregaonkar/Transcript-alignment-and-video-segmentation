import whisperx
import torch
import json


def transcribe_video(video_path, model_name="base", output_json="transcript.json"):
    """
    Transcribes a video file, displays transcript with timestamps, and saves it to a JSON file.

    :param video_path: Path to the video file.
    :param model_name: Whisper model size (e.g., "tiny", "base", "small", "medium", "large").
    :param output_json: Path to save the JSON output.
    :return: Transcript as a list of dictionaries (start, end, text).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    asr_options = {
        "multilingual": False,
        "hotwords": None,
    }
    
    print("Loading Whisper model...")
    model = whisperx.load_model(model_name, device=device, asr_options=asr_options)
    
    print("Transcribing the video...")
    result = model.transcribe(video_path)

    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    aligned_result = whisperx.align(result["segments"], model_a, metadata, video_path, device=device)

    transcript_data = []
    for seg in aligned_result["segments"]:
        transcript_data.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    print("Transcript with Timestamps:")
    for entry in transcript_data:
        print(f"[{entry['start']:.2f} - {entry['end']:.2f}]: {entry['text']}")

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(transcript_data, json_file, indent=4, ensure_ascii=False)
    
    print(f"Transcript saved to {output_json}")
    return transcript_data
    