def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")
