# src/generate.py
import numpy as np
import os
import music21 as m21
try:
    try:
        from keras.models import load_model
    except ImportError:
        from keras.models import load_model
except ImportError:
    from keras.models import load_model

def sample_with_temperature(preds, temperature=1.0):
    """
    Temperature sampling for probabilities array preds.
    """
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0:
        # argmax fallback
        return np.argmax(preds)
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sequence(model, int_to_note, note_to_int, seed_sequence, generate_length=200, temperature=1.0):
    """
    Generate a list of note tokens from a seed integer sequence.
    seed_sequence: list/np.array of length seq_length (integers).
    """
    seq_length = len(seed_sequence)
    pattern = seed_sequence.copy()
    generated = []
    n_vocab = len(note_to_int)
    for i in range(generate_length):
        x = np.array(pattern).reshape(1, seq_length)
        preds = model.predict(x, verbose=0)[0]
        index = sample_with_temperature(preds, temperature)
        result = int_to_note[index]
        generated.append(result)
        # move forward
        pattern = pattern[1:] + [index]
    return generated

def tokens_to_midi(token_sequence, output_path="output/generated_music.mid", tempo=120):
    """
    Convert token sequence to a music21 stream and save as MIDI.
    Tokens: either 'REST', pitch names like 'C4', or chord normalOrder encoded as '0.4.7' (depending on preprocess)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stream = m21.stream.Stream()
    # Set tempo
    mm = m21.tempo.MetronomeMark(number=tempo)
    stream.append(mm)
    for tok in token_sequence:
        if tok == "REST":
            r = m21.note.Rest()
            r.quarterLength = 0.5
            stream.append(r)
        elif '.' in tok and tok[0].isdigit():
            # chord from normalOrder indices (if your tokenization used normalOrder)
            try:
                pitches = [int(p) for p in tok.split('.')]
                chord_pitches = [m21.pitch.Pitch(m21.note.Note(p).pitch) for p in pitches]
                ch = m21.chord.Chord(pitches)
                ch.quarterLength = 0.5
                stream.append(ch)
            except Exception:
                # fallback treat as single note name
                try:
                    n = m21.note.Note(tok)
                    n.quarterLength = 0.5
                    stream.append(n)
                except Exception:
                    pass
        else:
            # assume pitch name like C4
            try:
                n = m21.note.Note(tok)
                n.quarterLength = 0.5
                stream.append(n)
            except Exception:
                # fallback: skip unknown token
                continue
    stream.write('midi', fp=output_path)
    return output_path

def load_and_generate(model_path, seed_sequence, int_to_note, note_to_int, generate_length=200, temperature=1.0, output_path="output/generated_music.mid"):
    """
    Load a saved model and generate a MIDI file.
    """
    model = load_model(model_path)
    generated_tokens = generate_sequence(model, int_to_note, note_to_int, seed_sequence, generate_length, temperature)
    midi_path = tokens_to_midi(generated_tokens, output_path)
    return midi_path
