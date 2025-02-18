# Personal Image Retriever

With the increasing volume of digital images, finding specific photos in large, unorganized collections can be a tedious task. This project addresses this problem by aiming to create a smart and intuitive image retrieval system that allows users to find images based on text queries and facial recognition. The system leverages multimodal models, vector stores, and semantic search techniques to provide accurate and efficient image retrieval.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Tools and Technologies](#tools-and-technologies)

## Features
- **Text-Based Search**: Find images by describing them in your own words.
- **Facial Recognition**: Locate images featuring specific individuals.
- **Efficient Retrieval**: Optimized indexing and ranking for quick results.
- **Scalable Design**: Built to handle large collections of images.

## How It Works

<img width="1000" alt="Architecture" src="https://github.com/user-attachments/assets/0f2946de-4393-4dbb-b441-d92d6e2f2213" />


1. **Preprocessing**: Images are processed to extract text descriptions and facial features.
2. **Indexing**: Extracted information is stored in separate indexes for quick retrieval.
3. **Query Execution**: Users enter a search query, which is analyzed for text and facial references.
4. **Matching**: The system compares the query with indexed data to find relevant images.
5. **Ranking**: Results are ranked based on similarity and displayed to the user.

## Folder Structure
Here is the structure of the project:

```plaintext
  <Personal Image Retriever>/
  │
  ├── experimentation/
  │   ├── 
  │   ├── 
  │   ├── 
  │   └── 
  │ 
  ├── application/   
  │   ├── 
  │   ├── 
  │   ├── 
  │ 
  ├── dataset/  
  │   ├── 
  │   ├── 
  │   ├── 
  │   ├── 
  │ 
  ├── requirements.txt
  └── README.md
```


## Tools and Technologies
* Language: Python
* Framework: PyTorch
* Multimodal Models: OpenAI CLIP, JINA CLIP V2
* Facial Detection Models: MTCNN, InceptionResnet V1
* Vector Store: Facebook AI Similarity Search (FAISS)
* Web Interface: Streamlit

