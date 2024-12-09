from databricks.connect import DatabricksSession
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow, time
import pandas as pd

def register_model(X_train, X_test, y_train, y_test,sample_input,model_name):
    try:
        mlflow.set_registry_uri("databricks-uc")
        with mlflow.start_run() as run:
            # Log the model with signature and input example
            signature = mlflow.models.infer_signature(
                model_input=X_train,
                model_output=clf.predict(X_train)
            )

            # Print the inferred signature
            print("\nInferred Model Signature:")
            print(signature)

            # Log the model with signature
            mlflow.sklearn.log_model(
                clf,
                "model",
                signature=signature,
                input_example=sample_input
            )

            # Make and log predictions
            predictions = clf.predict(X_test)
            accuracy = (predictions == y_test).mean()
            mlflow.log_metric("accuracy", accuracy)

            # Get run info
            print(f"MLflow Run ID: {run.info.run_id}")
            model_uri = f"runs:/{run.info.run_id}/model"
            print(f"Model URI: {model_uri}")

            # Register the model
            registered_model = mlflow.register_model(model_uri, model_name)
            print(f"Model registered as: {registered_model.name} version {registered_model.version}")

        time.sleep(10)
        # OPTIONAL STEP of making predictions
        print("\nPrediction Results:")
        print("------------------")

        results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions
        })

        print(results.head(5))
        print(f"\nModel Accuracy: {accuracy*100:.2f}%")
        run_id = run.info.run_id
        model_version = registered_model.version

    except Exception as e:
        print(str(e))
        run_id = 0
        model_version = 0
    finally:
        return (run_id, model_version, model_uri)

def predict(load_model_uri,sample_input):
    # Load the model with signature and make predictions
    loaded_model = mlflow.sklearn.load_model(load_model_uri)
    sample_prediction = loaded_model.predict(sample_input)
    print("\nSample Predictions using loaded model:")
    print(pd.DataFrame({
        'Input': [str(x) for x in sample_input.values],
        'Prediction': sample_prediction
    }))


if __name__ == "__main__":
    mlflow.set_registry_uri("databricks-uc")

    # Enable autologging
    mlflow.sklearn.autolog(
        log_model_signatures=True, # Enable automatic signature logging
        log_input_examples=True,   # Log input examples
        log_models=True            # Log model artifacts
    )

    # Load and split the data
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Number of rows in X_train : {X_train.shape[0]}")

    # Train the model
    clf = RandomForestClassifier(max_depth=7)
    clf.fit(X_train, y_train)

    # Create a sample input for signature inference
    sample_input = X_train.head(5)

    # Assign Model Name
    model_name = "gannychan.mc_demo.iris_model_with_sig_vscode_laptop"

    # Register the Model
    (run_id, model_version, model_uri) = register_model(X_train, X_test, y_train, y_test,sample_input,model_name)

    print("****************** Model Registered *******************")
    
    # Create another set of sample input for Prediction
    sample_input = X_train.sample(frac=0.2)
    load_model_uri = f"models:/{model_name}/{model_version}"
    
    # Predict (for Testing Purposess)
    # predict(load_model_uri, sample_input)