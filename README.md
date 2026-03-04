# GrapgRag_news

# RAG Q&A System

This project implements a simple Retrieval-Augmented Generation (RAG) system for answering legal questions using a small set of statutes and case law. It combines semantic search (FAISS) with a language model to generate answers grounded in provided legal sources.

## Overview

The system works in three main steps:

1. **Knowledge Storage**  
   Legal texts (statutes and case law) are manually defined in the `DOCS` list (as extra Knowledge). Each entry includes an `id` and the corresponding legal paragraph.

2. **Semantic Retrieval**  
   The model `sentence-transformers/all-MiniLM-L6-v2` converts each document into vector embeddings. These vectors are stored in a **FAISS index** to enable fast semantic similarity search.

3. **Answer Generation**  
   When a question is asked, the system retrieves the most relevant legal passages and sends them to a language model (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`).  
   The model generates an answer **using only the retrieved sources** and includes citations like `[1]`, `[2]`.

## How it Runs

1. The script builds a FAISS index from the legal documents.
2. The user enters a legal question.
3. The system retrieves the Top-K most relevant sources.
4. The language model generates a grounded answer with citations.

Run the program:

```bash
python app.py


### Input

Question (enter to quit): What is the definition of theft under the Theft Act 1968?


### Output

=== SOURCES USED ===
[1] score=0.734 TheftAct1968_s1
[2] score=0.692 TheftAct1968_s2
[3] score=0.541 Ivey_v_Genting_2017
[4] score=0.498 R_v_Ghosh_1982
[5] score=0.312 R_v_Smith_1974_Property

=== ANSWER ===

Under the Theft Act 1968, a person is guilty of theft if they dishonestly appropriate
property belonging to another with the intention of permanently depriving the other of it. [1]
