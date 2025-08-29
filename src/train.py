from src.preprocess import load_imdb_data
from src.model import build_rnn_model
import os

def train_model():
    (x_train, y_train), (x_test, y_test) = load_imdb_data()
    model = build_rnn_model()

    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/imdb_rnn.h5")
    print("✅ Model saved to models/imdb_rnn.h5")

    return model, (x_test, y_test)
