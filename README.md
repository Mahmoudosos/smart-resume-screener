# smart-resume-screener
An interactive AI-powered tool that compares resumes against job descriptions to measure fit and relevance. This project uses BERT-base and SBERT (Sentence-BERT) to compute semantic similarity between resumes and job postings.

## Features
- Upload your resume (PDF/DOCX).

- Upload multiple job descriptions (up to 5).

- Dual Model Comparison:
  - BERT-base: token embeddings + mean pooling.
  - SBERT: optimized for semantic similarity.

- Cosine similarity scores between resume and jobs.

- Visual plots for quick interpretation.

- Streamlit App – easy to use, interactive, and deployable online.

## Why this project??
Recruiters and job seekers often struggle to match resumes with job postings efficiently.
This app demonstrates how NLP & Transformers can:
- Speed up candidate screening.
- Help applicants identify their strongest matches.
- Provide insights into resume tailoring.

## Tech Stack:
- Python

- Streamlit

- PyTorch

- HuggingFace Transformers

- Sentence-Transformers

- Pandas

- Matplotlib


## How to Run:
Clone the repo, install dependencies, and run the app :
<pre>
  # Clone the repo 
  git clone https://github.com/your-username/resume-job-matcher.git cd resume-job-matcher 
  # Install dependencies 
  pip install -r requirements.txt 
  # Run the Streamlit 
  app streamlit run app.py </pre>

## Author

Mahmoud Osama – Aspiring Data Scientist | NLP Enthusiast | Open to Internships & Junior Roles

LinkedIn: https://www.linkedin.com/in/mahmoud-osama-52497a28b

Email: mahmoud.osama2021@feps.edu.eg

GitHub: https://github.com/Mahmoudosos 

  
