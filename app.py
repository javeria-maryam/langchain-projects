import streamlit as st
import requests

st.title("RAG Document Assistant")
st.write("Ask questions about your document and get grounded answers.")

question = st.text_input("Your question:", placeholder="What is machine learning?")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Answer:")
                    st.write(data["answer"])
                    st.info(f"Sources: {', '.join(data['sources'])}")
                else:
                    st.error("Something went wrong with the API response.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")