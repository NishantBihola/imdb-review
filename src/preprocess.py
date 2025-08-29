import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

def load_imdb_data(max_features=10000, max_len=200):
    print("📥 Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    # Pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    return (x_train, y_train), (x_test, y_test)
