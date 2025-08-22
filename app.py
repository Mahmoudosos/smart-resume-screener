import streamlit as st
import PyPDF2
import io
import os
import pandas as pd

st.title("Resume–Job Match Similarity")

# the resume uploading part:
uploaded_resume = st.file_uploader("Upload your resume", type=["pdf","docx"])
if uploaded_resume is not None:
    new_path = os.path.abspath("uploaded_resumes")
    st.write(new_path)
    os.makedirs(new_path,exist_ok=True)
    
    resume_path = os.path.join(new_path,uploaded_resume.name)
    # create the resume file
    with open(resume_path, "wb") as f:
        f.write(uploaded_resume.getbuffer())

# the descriptions uploading part
uploaded_descriptions = st.file_uploader("Upload your job descriptions",type=["pdf","docx"],accept_multiple_files=True)

# To make the max number of uploaded files is 5 
MAX_LINES = 5
if len(uploaded_descriptions) > MAX_LINES:

    st.warning(f"Maximum number of files reached. Only the first {MAX_LINES} will be processed.")

    uploaded_descriptions = uploaded_descriptions[:MAX_LINES]

if uploaded_descriptions is not None:
    new_path = os.path.abspath("uploaded_descriptions")
    job_descriptions_paths=[]
    job_descriptions_names= []
    os.makedirs(new_path,exist_ok=True)

    for i in uploaded_descriptions:
        save_path = os.path.join(new_path,i.name)
        job_descriptions_paths.append(save_path)
        job_descriptions_names.append(i.name)
        # create the job descriptions files
        with open(save_path, "wb") as f:
            f.write(i.getbuffer())
        

# The models part
col1, col2 = st.columns(2)

if  uploaded_resume and uploaded_descriptions:
    # Using BERT-Base model:
    with col1:
        st.subheader("Using BERT-Base Model:")
        # Uploading the functions from functions.py
        from functions import (clean_pdf,read_resume,inputs_gpu,bert_base_outputs,get_mean_pooling,get_cosine_similarity,draw_plot,run_bert_similarity)
        
        # call the functions with storing the cleaned text of resume and job descriptions 
        cleaned_text_resume,cleaned_text_job_descriptions = run_bert_similarity(resume_path,job_descriptions_paths,job_descriptions_names)

    with col2:
        # Using SBERT Model:
        st.subheader("Using SBERT Model:")

        # import the function from functions.py 
        from functions import sbert_model
        # call the function to get our cosine similarity scores
        df_smi = sbert_model(cleaned_text_resume,cleaned_text_job_descriptions)
        st.write(df_smi)
        # plot the cosine similarity scores
        draw_plot(x= job_descriptions_names, y= df_smi)
 