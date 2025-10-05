# SHL Assessment Recommendation Engine

This repository contains the code and experiments for building a recommendation engine for SHL assessments, featuring a web-based UI and a GET API endpoint. Below is an overview of the key files and directories.

## Files and Directories

- **`get_api_shl-main/`**  
  Contains the deployed code for a GET API hosted on Render.com, delivering JSON responses.  
  **URL:** `[https://get-api-shl.onrender.com/recommend](https://get-api-shl.onrender.com/recommend)`

- **`shl_web-main/`**  
  Houses the deployed code for a functional Streamlit framework application.  
  **URL:** `[https://testrecommendershl.streamlit.app](https://testrecommendershl.streamlit.app/)`

- **`SHL_Catalog_Extraction.ipynb`**  
  A Jupyter notebook with comprehensive web scraping code to extract the SHL product catalog from their website.

- **`SHL_Url_detail_extraction.ipynb`**  
  A Jupyter notebook featuring web scraping code to extract detailed information from individual assessment URLs.

- **`[Experiments] Building_the_RAG_pipeline.ipynb`**  
  An experimental Jupyter notebook documenting the development of the Retrieval-Augmented Generation (RAG) pipeline, including LLM-based preprocessing. (Needs to be downloaded, as it says invalid on GitHub) 

- **`[Experiments] Making_frontend.ipynb`**  
  A Jupyter notebook detailing local testing and development of the system’s frontend interface.

## Project Overview
This project scrapes SHL’s assessment catalog, preprocesses it with LLMs, builds a FAISS-based RAG pipeline, and deploys both a Streamlit UI and a FastAPI endpoint on Render.com. Refer to the 1-page report for a detailed approach.
