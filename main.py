from src.train import train_model
from src.evaluate import evaluate_model

def main():
    model, (x_test, y_test) = train_model()
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
