from app import train_model

def test_model():
    model = train_model()

    assert model is not None

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert len(prediction) == 1

    print("Test passed successfully!")
