import streamlit as st
from app.core.system import AutoMLSystem
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os

# Streamlit page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")
st.write("View and manage saved pipelines here.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Initialize pipeline list refresh trigger in session state
if 'refresh_pipelines' not in st.session_state:
    st.session_state.refresh_pipelines = False

# Retrieve saved pipelines
pipelines = automl.registry.list(type="pipeline")

if pipelines:
    # Select pipeline to load
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline_name = st.selectbox("Select a pipeline", pipeline_names)
    selected_pipeline = next(p for p in pipelines if
                             p.name == selected_pipeline_name)

    # Load pipeline data
    pipeline_data = pickle.loads(selected_pipeline.data)
    model = pickle.loads(pipeline_data["model"])

    # Retrieve the dataset used during training
    dataset_name = pipeline_data["dataset_name"]
    target_feature = pipeline_data["target_feature"]

    # Load the original dataset
    selected_dataset = next(d for d in automl.registry.list(type="dataset")
                            if d.name == dataset_name)
    dataset_data_bytes = selected_dataset.read()
    original_data = pd.read_csv(io.StringIO(dataset_data_bytes.decode()))

    # Get training values from the original dataset
    training_values = original_data[target_feature].values

    # Find and display selected pipeline details
    selected_pipeline = next(pipeline for pipeline in pipelines if
                             pipeline.name == selected_pipeline_name)

    st.write(f"**Pipeline Name:** {selected_pipeline.name}")
    st.write(f"**Version:** {selected_pipeline.version}")
    st.write(f"**Tags:** {', '.join(selected_pipeline.tags)}")

    # Delete Dataset Button
    if st.button("Delete Pipeline"):
        try:
            # Ensure dataset ID is valid before deletion
            if selected_pipeline.id:
                automl.registry.delete(selected_pipeline.id)
                st.success(f"Pipeline '{selected_pipeline.name}' deleted "
                           "successfully!")
                # Trigger pipeline list refresh
                st.session_state.refresh_pipeline = True
            else:
                st.error("Pipeline ID is missing. Unable to delete.")
        except Exception as e:
            st.error(f"Failed to delete pipeline: {e}")

    # Load pipeline data and display model details
    pipeline_data = pickle.loads(selected_pipeline.data)
    st.write("### Pipeline Details")

    # Deserialize model
    model = pickle.loads(pipeline_data["model"])
    is_trained = getattr(model, "trained", False)
    st.write("**Model Type:**", type(model).__name__)
    st.write("**Training Status:**", "Trained" if is_trained else
             "Not Trained")
    st.write("**Original Dataset:**", dataset_name)
    st.write("**Metrics:**")
    for metric_name, metric_value in pipeline_data["metrics"].items():
        st.write(f"- {metric_name}: {metric_value}")
    st.write("**Parameters:**")
    st.write("**Input Features:**", ", ".join(pipeline_data["input_features"]))
    st.write("**Target Feature:**", pipeline_data["target_feature"])
    st.write("**Training Split:**", f"{pipeline_data['train_split'] * 100}%")

    # # Display Model Attributes
    # if hasattr(model, "parameters") and model.parameters:
    #     st.write("**Model Parameters:**", model.parameters)
    # else:
    #     st.write("**Model Attributes:**")
    #     model_attributes = {k: v for k, v in model.__dict__.items() if not
    #                         k.startswith('_')}
    #     for key, value in model_attributes.items():
    #         st.write(f"{key}: {value}")

    # Prediction section
    st.header("Run Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions",
                                     type="csv")

    if uploaded_file:
        # Prepare data for predictions
        input_data = pd.read_csv(uploaded_file)
        input_features = pipeline_data["input_features"]
        required_features = [f for f in input_features if f in
                             input_data.columns]

        if len(required_features) < len(input_features):
            missing_features = set(input_features) - set(input_data.columns)
            st.error(f"Uploaded data is missing required features: {', '.join(
                missing_features)}")
        else:
            # Select the input columns needed for prediction
            input_features_data = input_data[required_features]

            # Generate predictions
            predictions = model.predict(input_features_data.values)
            input_data["Predictions"] = predictions

            # Plot training vs predicted values
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(training_values)), training_values,
                    label='Training',
                    marker='o', color='blue')
            ax.plot(range(len(predictions)), predictions, label='Predictions',
                    marker='x', color='red')
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title("Training vs. Predicted Values")
            ax.legend()
            st.pyplot(fig)

            # Provide download link for predictions as CSV
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv,
                               file_name="predictions.csv", mime="text/csv")

            # Save plot as image for the PDF report
            temp_image_path = "/tmp/predictions_plot.png"
            fig.savefig(temp_image_path)

            # Generate PDF report
            pdf = FPDF()
            pdf.add_page()

            pdf.add_font("ComicSans", "", "./assets/fonts/ComicSans.ttf",
                         uni=True)

            pdf.set_font("ComicSans", "", 16)
            pdf.cell(200, 10, txt="Model Prediction Report", ln=True,
                     align="C")

            # Add pipeline details
            pdf.set_font("ComicSans", size=11)
            pdf.cell(200, 10, txt=f"Pipeline: {selected_pipeline.name}",
                     ln=True)
            pdf.cell(200, 10, txt=f"Version: {selected_pipeline.version}",
                     ln=True)
            pdf.cell(200, 10, txt=f"Tags: {selected_pipeline.tags}",
                     ln=True)
            pdf.cell(200, 10, txt=f"Model type: {type(model).__name__}",
                     ln=True)
            pdf.cell(200, 10, txt=f"Original dataset: {dataset_name}",
                     ln=True)

            # Add metrics if available
            pdf.cell(200, 10, txt="Metrics:", ln=True)
            for metric_name, metric_value in pipeline_data["metrics"].items():
                pdf.cell(200, 10, txt=f"- {metric_name}: {metric_value}",
                         ln=True)

            pdf.cell(200, 10, txt=f"Input Features: {', '.join(pipeline_data[
                'input_features'])}", ln=True)
            pdf.cell(200, 10, txt=f"Target Feature: {pipeline_data[
                'target_feature']}", ln=True)
            pdf.cell(200, 10, txt=f"Training Split: {pipeline_data[
                'train_split'] * 100}%", ln=True)

            # Embed plot in PDF
            pdf.cell(200, 10, txt="Prediction plot:", ln=True)
            pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=180)

            # Output PDF to BytesIO
            pdf_output = io.BytesIO()
            pdf_content = pdf.output(dest="S").encode("latin1")
            pdf_output.write(pdf_content)
            pdf_output.seek(0)

            # Clean up temp image file
            os.remove(temp_image_path)

            # Download PDF
            st.download_button("Download PDF Report", data=pdf_output,
                               file_name="model_report.pdf",
                               mime="application/pdf")

else:
    st.write("No saved pipelines available.")
