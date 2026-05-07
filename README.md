# Adversarial Prompt Discovery in Large Language Models  
Authors:  
-	Jakob Garcia (Jgarcia713)  
-	Luis Cruz    (cruzhernandez778)  
-	Soren Abrams (siabrams)

Purpose:  
The goal of this project is to develop and refine techniques for generating arbitrary and novel strings with the GPT-2 LLM. This has implications for the security of LLMs, including with regards to prompt injections. The project consists of four parts:   
1. Core Implementation
	- This section includes functions necessary for the functionality of the later sections, including a test harness.
2. Manual Prompting and Technique Evaluation
	- This section explores a number of manual prompting techniques and analysis of their efficacy.
3. Automated Prompt Search
	- This section attempts to use a gradient descent model to generate prompts via machine learning.
4. Error Analysis
	- This section analyzes how effective the automated prompt search was at creating useful prompts.

How to Run:  
This project requires a Python 3 environment. In order to run this project, three files must be downloaded from the GitHub Repo:  
1. prompt.py
2. prompt.ipynb
3. requirements.txt

After downloading these files, the dependencies need to be installed. This can be done with:  
`pip install -r requirements.txt`  
Additionally, in order to run prompt.ipynb, Jupyter Notebook must be used. This can be installed via the Anaconda Navigator. 
