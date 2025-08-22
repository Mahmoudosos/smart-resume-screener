import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer,util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import re
import os
from transformers import AutoModel,AutoTokenizer

def clean_pdf(text:str) -> str:
  '''
  a function to remove unnecessery things like \n and \n\n and white spaces before and after the text.
  '''
  text = re.sub("(?<!\n)\n(?!\n)"," ",text)
  text = re.sub(r"\n{2:}","\n\n",text)
  text = re.sub("ï¼\u200b"," ",text)
  return text.strip()


def read_resume(file_path):
  '''
  a function to extract the text from the resume and put it in a txt form and clean this text afterwards
  '''
  text_path,text_extention = os.path.splitext(file_path)
  if "pdf" in text_extention:
    from pdfminer.high_level import extract_text
    text = extract_text(file_path)
    cleaned_text = clean_pdf(text)
  else:
    import docx
    text = docx.Document(file_path)
    text_list=[]
    for i in text.paragraphs:
      text_list.append(i.text)
    full_text = " ".join(text_list)
    cleaned_text=clean_pdf(full_text)
  return cleaned_text

def inputs_gpu(data,tokenizer):
  '''
  a funcation to turn your cells into tensors that holded in GPU

  '''
  inputs=[]
  inputs_gpu=[]
  for i in data:
    inputs.append(tokenizer( i,return_tensors="pt",truncation=True,padding=True)) 
  
  for i in inputs:
    inputs_gpu.append(i.to("cuda"))
  return inputs_gpu  

 
def bert_base_outputs(inputs,batch_size,tokenizer,model):

  '''
  This is a function to get your inputs to run it in the GPU, handling the 
  batch process, and return the last_hidden_states
  '''
  # import libraries
  all_embeddings = []
  all_attention_mask=[]
  
  for i in range(0, len(inputs), batch_size):
    batch_data = inputs[i:i+batch_size]
    
    # to remove the first dimension:
    input_ids_batch = [item['input_ids'].squeeze(0) for item in batch_data]
    attention_mask_batch = [item['attention_mask'].squeeze(0) for item in batch_data]
    
    # add padding. to make all the tensors with the same dimesnions:
    padded_input_ids = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
    

    # get the outputs:
    with torch.no_grad():
        outputs = model(
            input_ids=padded_input_ids.to("cuda"),
            attention_mask=padded_attention_mask.to("cuda")
        )
    # get the last_hidden_state for each tensor and append them into a list:
    last_hidden_states = outputs.last_hidden_state
    for i in range(0,last_hidden_states.size(0)):
      all_attention_mask.append(padded_attention_mask[i])
      
    for i in range(0,last_hidden_states.size(0)):
      all_embeddings.append(last_hidden_states[i])
  return all_embeddings,all_attention_mask

def get_mean_pooling(last_hidden_states,attention_mask):
  '''
  a function to get the mean_pooling from the last hidden states and attention maskes
  '''
  masked_embeddings = last_hidden_states * attention_mask
  sum_embeddings = masked_embeddings.sum(1)   
  token_counts = attention_mask.sum(1)         
  mean_pooled = sum_embeddings / torch.clamp(token_counts, min=1e-9)

  return mean_pooled

def get_cosine_similarity(tensor_1_list,tensor_2_list):
  '''
  a function to get the cosine similarity
  '''
  tensor_1 = torch.stack( [i.squeeze(0)for i in tensor_1_list])
  tensor_2 = torch.stack([i.squeeze(0) for i in tensor_2_list])

  # normalize the tensorers:
  tensor_1_norm = F.normalize(tensor_1)
  tensor_2_norm = F.normalize(tensor_2)

  # matrix multiplication for tensor_1_norm & tensor_2_norm
  cosine_matrix = torch.mm(tensor_1_norm,tensor_2_norm.T)
  df_cosine_matrix = pd.DataFrame(cosine_matrix.cpu().numpy())
  return df_cosine_matrix


    
def draw_plot(x,y):
  '''
  A function to draw the cosine_matrix
  '''
  fig,ax = plt.subplots(1,1)
  x_values = x
  y_values = list(np.round((y.loc[0].values)*100,2))
  chart_colors = ["#3674B5","#578FCA","#A1E3F9","#B2EBE0","#D1F8EF"]
  ax.bar(x= x,height=y_values,color = chart_colors[:len(x_values)])
  for i,s in zip(range(len(x_values)),y_values):
    ax.text(i,s,str(s)+"%",ha="center")
  ax.set_ylim(bottom=min(y_values)-20,top=100)
  ax.set_xticks(labels =x_values,ticks=x_values ,fontdict={"size":8,"rotation":45,"weight":"bold"})   
  ax.set_ylabel("% of similarity")
  ax.set_xlabel("The job descriptions")
  st.pyplot(fig)

def get_jobtitle_descriptions_indexes_and_percentages(id,n_largest,similarity_matrix,df_resumes,df_job_description):
  id_index= df_resumes[df_resumes["ID"]==id].index.item() # get the index of the id
  top_descriptions= similarity_matrix.loc[id_index].astype(float).nlargest(n_largest) # get the top_descriptions descripations of this id
  job_descriptions_indexes = list(similarity_matrix.loc[id_index].astype(float).nlargest(n_largest).index) # get the indexes of the top descripations
  job_title = [df_job_description.loc[i,"Job Title"] for i in job_descriptions_indexes ] # get the job title of the top 5 descripations
  description_similarity_percent = list(top_descriptions)
  return(job_descriptions_indexes,job_title,description_similarity_percent)

def top_n_descriptions(similarity_matrix, df_resumes, df_job_description): 
    ''' 
    A function to get the top n descriptions and print them in a meaningful way 
    ''' 
    i = 0 
    while True: 
        try: 
            id = int(input("please enter your id:")) 
            n_largest = int(input("please enter the number of top descriptions:")) 
        except ValueError: 
            print("invalid input! please enter your id:") 
            continue  # goes back to the top of the loop 
 
        if id in list(df_resumes["ID"]):  # check if the input id in our list 
            category = df_resumes["Category"][df_resumes["ID"] == id].item()  
            print(f"Welcome {id}, your Category is {category}\n The Top {n_largest} job descriptions for your resume are:\n") 
            
            job_descriptions_indexes, job_title, description_similarity_percent = get_jobtitle_descriptions_indexes_and_percentages(
                id=id,
                n_largest=n_largest,
                similarity_matrix=similarity_matrix,
                df_resumes=df_resumes,
                df_job_description=df_job_description
            ) 
 
            for i in range(n_largest): 
                title_index = f"{job_title[i]}_{job_descriptions_indexes[i]}" 
                print(f"{i+1}. {title_index:35} --> with {description_similarity_percent[i]*100:0.2f}%") 
            break 
        else: 
            i += 1  
            print("Sorry! this is an incorrect ID") 
      
            if i == 5: 
                print("please, try again later") 
                break



def draw_top_n_descriptions(similarity_matrix,df_resumes,df_job_description):

  '''
  a funation to draw the top n descriptions using matplotlib
  '''
  i=0
  while True:
    try:
      id = int(input("please enter your id:"))
      n_largest= int(input("please enter the number of top descriptions:"))

    except(ValueError):
      i+=1
      if i<=5:
        print("Invalid input! please enter your id in a numeric form")
        continue
      else:
        print("please try again later")
        break

    job_descriptions_indexes,job_title ,description_similarity_percent=get_jobtitle_descriptions_indexes_and_percentages(id=id,n_largest=n_largest,similarity_matrix=similarity_matrix,df_resumes=df_resumes,df_job_description=df_job_description)

    # get the x-axis values
    x_values= []
    for i in range(n_largest):
      x_values.append((job_title[i])+("_")+ str(job_descriptions_indexes[i]))
    
    # get the y-axis values
    y_values = [round(i*100,2) for i in description_similarity_percent]

    chart_colors=["#3674B5","#578FCA","#A1E3F9","#B2EBE0","#D1F8EF"]
    fig,axis=plt.subplots()
    
    axis.bar(x=x_values , height=y_values,color=chart_colors[:n_largest])
    
    axis.set_xticks(labels =x_values,ticks=x_values ,fontdict={"size":8,"rotation":45,"weight":"bold"})
    
    axis.set_ylim(bottom=min(y_values)-20,top=100)
    
    # to show the percentages of our x-axis labels
    for i,s in zip(y_values,range(n_largest)):
      axis.text(s,i,str(i)+"%",ha="center")
    
    axis.set_title(f"Top {n_largest} descriptions that matches your resume",pad=10)
    axis.set_xlabel("job_index")
    axis.set_ylabel(" % of similarity")
    plt.show
    break
    
def sbert_model(resume_text, job_descriptions_text):
    '''
    A function to build a SBERT model.
    '''
    # Encode all texts in one go (batching)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embeddings_resume = model.encode(resume_text, convert_to_tensor=True, batch_size=8, device=device)
    embeddings_job_descriptions = model.encode(job_descriptions_text, convert_to_tensor=True, batch_size=8,device=device)

    # Compute cosine similarity for all pairs
    cos_sim_matrix = util.cos_sim(embeddings_resume, embeddings_job_descriptions).cpu().numpy()

    # Put in a dataframe
    df_cos_sim = pd.DataFrame(cos_sim_matrix, index=range(len(resume_text)), columns=range(len(job_descriptions_text)))
    
    return df_cos_sim 

def run_bert_similarity(resume_path,job_descriptions_paths,job_descriptions_names):
    '''
    A function that containes the different functions to build and draw the the bert similarity
    '''
    model_name = "google-bert/bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda").to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cleaned_text_resume = [read_resume(resume_path)]
    cleaned_text_job_descriptions = []
    for i in job_descriptions_paths:
        cleaned_text_job_descriptions.append(read_resume(i))

    inputs_job_descriptions_gpu = inputs_gpu(cleaned_text_job_descriptions,tokenizer=tokenizer)
    # Get the inputs :
    inputs_resume_gpu = inputs_gpu(cleaned_text_resume,tokenizer=tokenizer)

    
    # get the last_hidden_state and padded_attention for our resume data
    last_hidden_states_resume,padded_attention_mask_resume = bert_base_outputs(inputs=inputs_resume_gpu ,batch_size=1,tokenizer=tokenizer,model=model)
    # get the last_hidden_state and padded_attention for our job_descriptions data
    last_hidden_states_descriptions,padded_attention_mask_descriptions = bert_base_outputs(inputs=inputs_job_descriptions_gpu,batch_size=1,tokenizer=tokenizer,model=model)

    
    # fix the size of resume tensor
    last_hidden_states_resume = pad_sequence(last_hidden_states_resume,batch_first=True)
    padded_attention_mask_resume= pad_sequence(padded_attention_mask_resume,batch_first=True)
    padded_attention_mask_resume = padded_attention_mask_resume.unsqueeze(-1) 
    # fix the size of job_descriptions tensor
    last_hidden_states_descriptions = pad_sequence(last_hidden_states_descriptions,batch_first=True)
    padded_attention_mask_descriptions=pad_sequence(padded_attention_mask_descriptions,batch_first=True)
    padded_attention_mask_descriptions = padded_attention_mask_descriptions.unsqueeze(-1) 
    
    # get the mean pooling of resume and job descriptions
    resume_mean_embeddings = get_mean_pooling(last_hidden_states_resume,padded_attention_mask_resume)
    description_mean_embeddings = get_mean_pooling(last_hidden_states_descriptions,padded_attention_mask_descriptions)
    
    # get the cosine matrix
    cosine_matrix = get_cosine_similarity(resume_mean_embeddings,description_mean_embeddings)
    st.write(cosine_matrix)
    
    # draw the cosine_matrix
    draw_plot(x=job_descriptions_names,y=cosine_matrix)
    return(cleaned_text_resume,cleaned_text_job_descriptions)