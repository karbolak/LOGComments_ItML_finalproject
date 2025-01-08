import streamlit as st
import pandas as pd
import pickle
import io

from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import check_multicollinearity
from autoop.core.ml.model import (get_model, REGRESSION_MODELS,
                                  CLASSIFICATION_MODELS)
from autoop.core.ml.metric import (REGRESSION_METRICS, CLASSIFICATION_METRICS,
                                   LOG_CLASSIFICATION_METRICS, get_metric)
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.pipeline import Pipeline


# Streamlit page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Retrieve available datasets
datasets = automl.registry.list(type="dataset")

# Initialize session states for button flags
if 'train_button_flag' not in st.session_state:
    st.session_state.train_button_flag = False
if 'save_button_flag' not in st.session_state:
    st.session_state.save_button_flag = False


# Helper function for styling text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# Wrapper for DataFrame to mimic Dataset with a read method
class DatasetWrapper:
    def __init__(self, data_frame):
        self._data_frame = data_frame

    def read(self):
        return self._data_frame


st.write("# ⚙ Modelling")
write_helper_text("Design a machine learning pipeline to train a model on a "
                  "dataset.")

if datasets:
    # Dataset Selection
    st.header("1. Select a Dataset")
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Choose a dataset for training",
                                         dataset_names)

    # Retrieve selected dataset and convert from bytes to DataFrame
    selected_dataset = next(dataset for dataset in datasets if
                            dataset.name == selected_dataset_name)
    dataset_data_bytes = selected_dataset.read()
    dataset_df = pd.read_csv(io.StringIO(dataset_data_bytes.decode()))
    st.write("### Dataset Preview")
    st.write(dataset_df.head())

    # Wrap dataset in DatasetWrapper
    dataset_wrapped = DatasetWrapper(dataset_df)

    # Feature selection and task type detection
    st.header("2. Select Features")
    features = detect_feature_types(dataset_df)
    feature_names = [feature.name for feature in features]
    selected_target_feature = st.selectbox("Select target feature",
                                           feature_names)
    available_input_features = [name for name in feature_names if
                                name != selected_target_feature]
    selected_input_features = st.multiselect("Select input features",
                                             available_input_features)

    target_feature = next((f for f in features if
                           f.name == selected_target_feature), None)
    task_type = ("classification" if target_feature and
                 target_feature.type == "categorical" else "regression")
    st.write(f"**Detected Task Type:** {task_type}")

    # Model and metric selection
    st.header("3. Select a Model")
    if task_type == "classification":
        ALT_CLASSIFICATION_MODELS = CLASSIFICATION_MODELS
        # Check if the target feature contains only 2 unique values
        if len(set(dataset_df[selected_target_feature])) != 2:
            # Remove logistic regression from the classification models
            ALT_CLASSIFICATION_MODELS = [model for model in
                                         CLASSIFICATION_MODELS if model !=
                                         "LogisticRegression"]
            st.info("Logistic Regression requires a binary target feature. "
                    "It has been removed from the model options.")
    else:
        REG_MODELS = REGRESSION_MODELS
        # Check for multicollinearity
        subsection_df = dataset_df[selected_input_features]
        if len(selected_input_features) < 2:
            st.warning("Please select more than one feature to proceed with "
                       "modeling.")
        else:
            if check_multicollinearity(subsection_df):
                # Use Ridge regression instead of multiple linear regression
                # in case of multicollinearity
                REG_MODELS = [model for model in REG_MODELS if model !=
                              "MultipleLinearRegression"]
                st.info("Multicollinearity detected. Ridge regression is "
                        "recommended over multiple linear regression.")
            else:
                REG_MODELS = [model for model in REG_MODELS if model !=
                              "RidgeRegression"]
                st.info("No multicollinearity detected. Multiple linear "
                        "regression is recommended over ridge regression.")
    model_choices = (REG_MODELS if task_type == "regression"
                     else ALT_CLASSIFICATION_MODELS)
    selected_model_name = st.selectbox("Choose a model", model_choices)
    # te ify chyba powinny być tutaj?
    selected_model = get_model(selected_model_name)

    if selected_model != "MultipleLinearRegression":

        st.header("3.5 Select parameters for the ML model:")

        # Define parameter selection based on the selected model
        if selected_model == "KNearestNeighbors":
            # Allow the user to choose the value of k for K-Nearest Neighbors
            k = st.number_input("Select value of k:", min_value=1, step=1)
            st.write("You selected k =", k)
            # Pass k to your function as needed here

        elif selected_model == "RidgeRegression":
            # Allow the user to choose alpha for Ridge Regression
            alpha = st.number_input("Select alpha value:", min_value=0.01,
                                    step=0.01)
            st.write("You selected alpha =", alpha)
            # Pass alpha to your function as needed here

        elif selected_model == "LogisticRegression":
            # Check if params_from_last_col is an integer and between 0 and 1
            params_from_last_col = st.text_input("Enter a parameter value (0"
                                                 " or 1):", value="0")
            if not (params_from_last_col.isdigit() and
                    int(params_from_last_col) in [0, 1]):
                st.warning("Please enter 0 or 1.")
                choose_label_btn = st.button("Choose labels (0 and 1)")
                if choose_label_btn:
                    st.write("Label selection button clicked.")
                    # we have to sub for all values here and pass the changed
                    # values into the later parts of the code

            # Allow the user to choose learning rate and number of iterations
            learning_rate = st.number_input("Select learning rate:",
                                            min_value=0.0001, max_value=1.0,
                                            step=0.0001, format="%.4f")
            n_iterations = st.number_input("Select number of iterations:",
                                           min_value=1, step=1)
            st.write(f"You selected learning rate = {learning_rate} and "
                     f"iterations = {n_iterations}")
            # Pass learning_rate and n_iterations to your function here

        elif selected_model in ["DecisionTreeRegression",
                                "DecisionTreeClassification"]:
            # Allow the user to choose max_depth for Tree-based models
            max_depth = st.number_input("Select max depth:", min_value=1,
                                        step=1)
            st.write("You selected max_depth =", max_depth)
            # Pass max_depth to your function as needed here

    st.header("4. Select Dataset Split")
    train_split = st.slider("Training Set Split (%)", min_value=50,
                            max_value=90, value=80, step=5)
    test_split = 100 - train_split

    st.header("5. Select Metrics")

    if selected_model_name == "LogisticRegression":
        available_metrics = LOG_CLASSIFICATION_METRICS
    elif task_type == "regression":
        available_metrics = REGRESSION_METRICS
    else:
        available_metrics = CLASSIFICATION_METRICS

    selected_metrics = st.multiselect("Choose metrics", available_metrics)
    metric_objects = [get_metric(metric_name) for metric_name in
                      selected_metrics]

    # Check if all required fields are complete
    is_ready_to_train = all([
        selected_dataset_name,  # Dataset selected
        selected_target_feature,  # Target feature selected
        selected_input_features,  # At least one input feature selected
        selected_model_name,  # Model selected
        selected_metrics  # At least one metric selected
    ])

    # Display Pipeline Summary
    st.header("6. Pipeline Summary")
    st.write("### Configuration Summary")
    st.write(f"**Selected Dataset:** {selected_dataset_name}")
    st.write(f"**Selected Target Feature:** {selected_target_feature}")
    st.write(f"**Selected Input Features:** {', '.join(
        selected_input_features)}")
    st.write(f"**Selected Model:** {selected_model_name}")
    st.write(f"**Training Split:** {train_split}% | **Testing Split:** "
             f"{test_split}%")
    st.write(f"**Selected Metrics:** {', '.join(selected_metrics)}")

    # Display warning if required fields are missing
    if not is_ready_to_train:
        st.warning("Please complete all selections (dataset, target feature, "
                   "input features, model, and metrics) to enable training.")

    # Train Model Button
    if st.button("Train Model", disabled=not is_ready_to_train):
        st.session_state.train_button_flag = True

    # Check if training is complete and show metrics/results
    if st.session_state.train_button_flag and is_ready_to_train:
        # Initialize the pipeline with the wrapped dataset
        pipeline = Pipeline(
            metrics=metric_objects,
            dataset=dataset_wrapped,  # Use wrapped dataset
            model=selected_model,
            input_features=[f for f in features if f.name in
                            selected_input_features],
            target_feature=target_feature,
            split=train_split / 100
        )

        results = pipeline.execute()
        trained_model = results["trained_model"]
        # Assuming this retrieves true train labels
        train_Y = results.get("train_Y")

        # Verify model is trained
        if not getattr(trained_model, "trained", False):
            st.warning("Model training failed.")
        else:
            st.success("Model training completed!")

        # Display Evaluation Metrics
        evaluation_metrics = results.get("metrics", [])
        if evaluation_metrics:
            st.write("### Evaluation Metrics")
            for metric_name, metric_value in evaluation_metrics:
                st.write(f"{metric_name}: {metric_value}")
        else:
            st.write("No evaluation metrics available.")

        metrics_with_values = {metric_name: metric_value for metric_name,
                               metric_value in evaluation_metrics}

        # Pipeline saving section
        st.header("7. Save Pipeline")
        pipeline_name = st.text_input("Enter a name for the pipeline",
                                      value="Pipeline")
        pipeline_version = st.text_input("Enter pipeline version",
                                         value="1.0.0")

        # Check if the pipeline configuration is complete
        is_ready = all([
            selected_input_features,
            selected_target_feature,
            selected_metrics,
            pipeline_name
        ])

        # Save Pipeline button
        if st.button("Save Pipeline", disabled=not is_ready):
            st.session_state.save_button_flag = True

    # Check if Save Pipeline is triggered and complete saving
    if st.session_state.save_button_flag and is_ready:
        # Prepare pipeline data for serialization
        pipeline_data = {
            "model": pickle.dumps(trained_model),  # Serialize trained model
            "metrics": metrics_with_values,
            "input_features": selected_input_features,
            "target_feature": selected_target_feature,
            "train_split": train_split / 100,
            "dataset_name": selected_dataset_name,
        }

        # Create artifact for pipeline
        pipeline_artifact = Artifact(
            name=pipeline_name,
            asset_path=f"pipelines/{pipeline_name}.pkl",
            data=pickle.dumps(pipeline_data),
            version=pipeline_version,
            type="pipeline",
            tags=["pipeline", task_type],
        )

        # Register the pipeline in the artifact registry
        automl.registry.register(pipeline_artifact)
        st.success(f"Pipeline '{pipeline_name}' (v{pipeline_version}) saved "
                   "successfully!")
        st.session_state.train_button_flag = False  # Reset after save
        st.session_state.save_button_flag = False
else:
    st.write("No datasets available.")
