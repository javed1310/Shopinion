import streamlit as st
import torch
import shap
import pickle
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import matplotlib.pyplot as plt # --- ADD THIS LINE ---

# --- App Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis with XAI",
    page_icon="🧠",
    layout="wide"
)

# --- Caching: Load assets once and reuse ---
@st.cache_resource
def load_assets():
    """
    Loads the fine-tuned model, tokenizer, and label encoder from disk.
    This function is cached to ensure it runs only once.
    """
    save_directory = "C:/Users/javed/Downloads/model"
    label_encoder_path = "C:/Users/javed/Downloads/label_encoder.pkl"

    st.info("Loading model assets... This may take a moment.")
    
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Add label mappings to the model's configuration
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    model.config.id2label = id2label
    model.config.label2id = label2id
        
    st.success("✅ Model assets loaded successfully!")
    return model, tokenizer, label_encoder

# --- Load all the required assets ---
model, tokenizer, label_encoder = load_assets()

# --- Create the explainer and pipeline (also cached) ---
@st.cache_resource
def get_explainer_and_pipeline(_model, _tokenizer):
    """
    Creates and caches the SHAP explainer and the Transformers pipeline.
    """
    sentiment_pipeline = pipeline(
        "text-classification",
        model=_model,
        tokenizer=_tokenizer,
        return_all_scores=True
    )
    
    explainer = shap.Explainer(sentiment_pipeline)
    return explainer, sentiment_pipeline

explainer, sentiment_pipeline = get_explainer_and_pipeline(model, tokenizer)


# --- Streamlit User Interface ---
st.title("🧠 Sentiment Analysis with Explainable AI (XAI)")
st.markdown(
    "This app uses a fine-tuned **DistilBERT** model to predict sentiment and **SHAP** to explain the prediction. "
    "Enter a sentence below to see how it works!"
)

user_input = st.text_area(
    "Enter a sentence for analysis:", 
    "The new design is sleek and modern, but I'm worried about the price."
)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            prediction = sentiment_pipeline(user_input)[0]
            best_prediction = max(prediction, key=lambda x: x['score'])
            predicted_sentiment = best_prediction['label']
            confidence_score = best_prediction['score']

        st.subheader("✅ Prediction Result")
        st.success(f"**The predicted sentiment is: `{predicted_sentiment}`** (Confidence: {confidence_score:.2%})")
        
        with st.spinner("Generating explanations..."):
            st.subheader("🎨 Explanation of the Prediction (SHAP Values)")
            st.markdown(
                "The plots below show how each feature contributed to the model's output value for each class."
            )
            
            shap_values = explainer([user_input])

            class_names = label_encoder.classes_.tolist()
            for i, class_name in enumerate(class_names):
                st.write(f"**Explanation for the '{class_name}' class:**")
                
                # Use shap.waterfall_plot and display it with st.pyplot
                shap.waterfall_plot(shap_values[0, :, i], show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf() # Clear the plot, crucial for loops
    else:
        st.warning("Please enter a sentence to analyze.")