from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vectorStore import vectorstore

import streamlit as st

st.set_page_config(page_title='CVBot', page_icon='🧠', layout='centered')
st.title('CVBot - Assistant for CV-based Q&A')

# ===== SIDEBAR =====
st.sidebar.header('Model & Retrieval Settings')

temperature = st.sidebar.slider(
    'Model temperature (creativity):',
    min_value=0.0, max_value=1.0, value=0.1, step=0.05,
    help='Higher values make the model more creative, lower values make it more deterministic.'
)

k_value = st.sidebar.slider(
    'Number of CV chunks to retrieve (k):',
    min_value=1, max_value=10, value=3, step=1,
    help='How many relevant CV sections to use as information to answer.'
)

@st.cache_resource
def load_model(temperature: float):
    model = OllamaLLM(model='llama3.1', temperature=temperature)
    return model

model = load_model(temperature=temperature)

with st.expander('Add manual context (optional)'):
    manual_context = st.text_area(
        'Add any extra context you want the model to know before reading your CV:',
        placeholder='Example: This CV belongs to a software engineer specialized in data science...',
        height=150
    )

question = st.text_input('Ask your question to the CVBot:')

if st.button('Ask CVBot'):
    if not question.strip():
        st.warning('Please write a question first.')
    else:
        base_context = """
        Behave like you are the person from the CV below:

        CV (curriculum viate): {CV}

        Respond the question below, and treat me with 'Sir'. Be brief and concise.

        Question: {question}
        """

        if manual_context and manual_context.lower != 'no':
            context = manual_context + "\n" + base_context
        else:
            context = base_context

    template = PromptTemplate.from_template(context)

    with st.spinner('Searching CV and generating answer...'):
        relevant_documentation = vectorstore.similarity_search(query=question, k=k_value)
        cv = "\n\n".join([doc.page_content for doc in relevant_documentation])
        prompt = template.invoke(
            {
                'CV': cv,
                'question': question
            }
        )
        answer = model.invoke(prompt)
    
    st.subheader("CVBot's Answer:")
    st.success(answer)