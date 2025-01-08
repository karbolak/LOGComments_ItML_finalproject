import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Initialize dataset list refresh trigger in session state
if 'refresh_datasets' not in st.session_state:
    st.session_state.refresh_datasets = False

# Page title
st.title("📊 Datasets")

st.write("Use this page to upload, view, manage, and delete datasets in the "
         "AutoML system.")

# Section: Display Existing Datasets
st.header("Available Datasets")

# Check session state to determine if datasets need to be reloaded
if st.session_state.refresh_datasets:
    datasets = automl.registry.list(type="dataset")
    st.session_state.refresh_datasets = False  # Reset the refresh flag
else:
    datasets = automl.registry.list(type="dataset")

if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset to view",
                                         dataset_names)

    # Find selected dataset and display details
    selected_dataset = next(dataset for dataset in datasets if
                            dataset.name == selected_dataset_name)
    st.write(f"**Name:** {selected_dataset.name}")
    st.write(f"**Path:** {selected_dataset.asset_path}")
    st.write(f"**Version:** {selected_dataset.version}")
    st.write(f"**Tags:** {', '.join(selected_dataset.tags)}")

    # Display dataset preview
    st.write("### Dataset Preview")
    try:
        dataset_data = selected_dataset.read()
        # Convert bytes to DataFrame
        df = pd.read_csv(io.StringIO(dataset_data.decode()))
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading dataset: {e}")

    # Delete Dataset Button
    if st.button("Delete Dataset"):
        try:
            # Ensure dataset ID is valid before deletion
            if selected_dataset.id:
                automl.registry.delete(selected_dataset.id)
                st.success(f"Dataset '{selected_dataset.name}' deleted "
                           "successfully!")
                # Trigger dataset list refresh
                st.session_state.refresh_datasets = True
            else:
                st.error("Dataset ID is missing. Unable to delete.")
        except Exception as e:
            st.error(f"Failed to delete dataset: {e}")
else:
    st.write("No datasets available.")

# Section: Upload New Dataset
st.header("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

if uploaded_file:
    # Load CSV data and show a preview
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.write(data.head())

    # Dataset details
    dataset_name = st.text_input("Enter a name for the dataset",
                                 value="dataset")
    dataset_version = st.text_input("Enter dataset version", value="1.0.0")
    tags = st.text_input("Tags (comma-separated)", value="")

    # Button to save dataset
    if st.button("Save Dataset"):
        try:
            # Parse and tags
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            # Create Dataset object using from_dataframe
            dataset = Dataset.from_dataframe(
                data=data,
                name=dataset_name,
                asset_path=f"datasets/{dataset_name}.csv",
                version=dataset_version
            )
            dataset.tags = tags_list

            # Register dataset in the artifact registry
            automl.registry.register(dataset)
            st.success(f"Dataset '{dataset_name}' saved successfully!")
            # Set session state to trigger dataset list refresh
            st.session_state.refresh_datasets = True
        except Exception as e:
            st.error(f"Failed to save dataset: {e}")

# Button to refresh datasets
if st.button("Refresh Dataset List"):
    # Set session state to trigger dataset list refresh
    st.session_state.refresh_datasets = True
