import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference

# Watsonx credentials and settings
model_id = "ibm/granite-3-8b-instruct"
project_id = "e148ca84-35e1-433d-9d5e-a71c64c3def8"
credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com",
    "apikey": "bCDB66qGQ4GEdDAu6o6kQ-BM4iLenxHfXZDZCrwtMwKf"
}

# Streamlit UI
st.title("EduTutor AI")

question = st.text_input("Ask your question:")

if st.button("Get Answer") and question.strip() != "":
    model = ModelInference(
        model_id=model_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 500
        },
        project_id=project_id,
        credentials=credentials
    )

    response = model.generate(question)
    answer = response["results"][0]["generated_text"]
    st.write(answer)
