# main.py
import random
import numpy as np
from src.data_loader import list_midi_files
from src.preprocess import parse_midi, extract_notes_from_stream, prepare_sequences
from src.train import train_model
from src.generate import load_and_generate

DATA_DIR = "data/midi_files"
SEQ_LENGTH = 50
EPOCHS = 30   # lower for quick tests
BATCH_SIZE = 64
MODEL_PATH = "models/trained_model.h5"

def build_dataset():
    midi_files = list_midi_files(DATA_DIR)
    print(f"Found {len(midi_files)} MIDI files.")
    all_notes = []
    for i, mfile in enumerate(midi_files):
        print(f"Parsing {i+1}/{len(midi_files)}: {mfile}")
        s = parse_midi(mfile)
        tokens = extract_notes_from_stream(s)
        all_notes.extend(tokens)
    return all_notes

def main():
    notes = build_dataset()
    if len(notes) < SEQ_LENGTH + 1:
        raise ValueError("Not enough notes to create sequences. Add more MIDI files or reduce SEQ_LENGTH.")
    X, y, note_to_int, int_to_note, n_vocab = prepare_sequences(notes, seq_length=SEQ_LENGTH)
    print("Prepared sequences:", X.shape, y.shape, "Vocab size:", n_vocab)
    # Train model
    model = train_model(X, y, n_vocab=n_vocab, seq_length=SEQ_LENGTH, model_save_path=MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # Example: pick a random seed from X and generate
    seed_idx = random.randint(0, len(X)-1)
    seed_sequence = X[seed_idx].tolist()
    print("Using seed sequence index:", seed_idx)
    output_midi = load_and_generate(MODEL_PATH, seed_sequence, int_to_note, note_to_int, generate_length=200, temperature=1.0, output_path="output/generated_music.mid")
    print("Generated MIDI saved to:", output_midi)

if __name__ == "__main__":
    main()