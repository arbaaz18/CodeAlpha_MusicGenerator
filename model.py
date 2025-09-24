# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.utils import to_categorical

def build_lstm_model(seq_length: int, n_vocab: int, embed_dim: int = 100, lstm_units: int = 256):
    """
    Return a compiled Keras LSTM model for next-token prediction.
    Input: sequences of integers (seq_length,)
    Output: softmax over n_vocab tokens.
    """
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab, output_dim=embed_dim, input_length=seq_length))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(lstm_units))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
