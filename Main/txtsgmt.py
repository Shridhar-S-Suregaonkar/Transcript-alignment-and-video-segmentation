import json
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch

def load_transcript(json_path):
    """
    Load the transcript data from a JSON file.
    
    :param json_path: Path to the JSON file containing the transcript.
    :return: List of dictionaries with 'start', 'end', and 'text' keys.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        transcript_data = json.load(file)
    return transcript_data

def sentence_encoder(transcript_data, model_name="bert-base-uncased", device="cpu"):
    """
    Encodes each sentence into a fixed-size embedding using a BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    
    sentences = [entry["text"] for entry in transcript_data]
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(sentence_embedding)
    
    return np.array(embeddings)

def paragraph_encoder(sentence_embeddings, cluster_labels):
    """
    Encodes paragraphs by aggregating sentence embeddings within the same cluster.
    """
    paragraphs = {}
    for idx, label in enumerate(cluster_labels):
        if label not in paragraphs:
            paragraphs[label] = []
        paragraphs[label].append(sentence_embeddings[idx])
    
    paragraph_embeddings = {label: np.mean(embeddings, axis=0) for label, embeddings in paragraphs.items()}
    return paragraph_embeddings

def cluster_transcript(transcript_data, sentence_embeddings, n_clusters=None, distance_threshold=1.5):

    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        distance_threshold=distance_threshold, 
        linkage="ward"
    )
    labels = clustering_model.fit_predict(sentence_embeddings)

    clustered_segments = {}
    for idx, label in enumerate(labels):
        if label not in clustered_segments:
            clustered_segments[label] = {
                "start": transcript_data[idx]["start"],
                "end": transcript_data[idx]["end"],
                "text": [transcript_data[idx]["text"]]
            }
        else:
            clustered_segments[label]["end"] = transcript_data[idx]["end"]
            clustered_segments[label]["text"].append(transcript_data[idx]["text"])
    
    return clustered_segments, labels

def textual_segmentation(json_path, output_json="segmented_transcript.json", model_name="bert-base-uncased"):

    print("Loading transcript...")
    transcript_data = load_transcript(json_path)

    print("Performing sentence-level encoding...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_embeddings = sentence_encoder(transcript_data, model_name=model_name, device=device)

    print("Clustering transcript into segments...")
    clustered_segments, cluster_labels = cluster_transcript(transcript_data, sentence_embeddings)

    print("Performing paragraph-level encoding...")
    paragraph_embeddings = paragraph_encoder(sentence_embeddings, cluster_labels)

    formatted_segments = []
    for cluster_id, segment in clustered_segments.items():
        formatted_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": " ".join(segment["text"]),
            "paragraph_embedding": paragraph_embeddings[cluster_id].tolist()
        })

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(formatted_segments, json_file, indent=4, ensure_ascii=False)
    
    print(f"Segmented transcript saved to {output_json}")
    return formatted_segments

def display_segments(json_path):

    with open(json_path, "r", encoding="utf-8") as file:
        segmented_data = json.load(file)

    print("Segments:")
    for idx, segment in enumerate(segmented_data, 1):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        print(f"Segment {idx}:")
        print(f"  Start Time: {start_time:.2f}")
        print(f"  End Time: {end_time:.2f}")
        print(f"  Text: {text}\n")

    print(f"Total number of segments: {len(segmented_data)}")