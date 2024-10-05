# FT_llama_3.1

![Designer](https://github.com/user-attachments/assets/f89c473c-a3cf-404f-a7ef-9945e6050578)

**What is Llama 3.1?**

Llama 3.1 is the newer set of the Llama family of models trained and released recently by the Meta Organization. Meta has released 8 models with 3 base version models and 5 finetuned version models. The three base models include Llama 3.1 8B, Llama 3.1 70B, and the newly introduced and state-of-the-art open-source model Llama 3.1 405B. 

**Why should we fine-tune Llama 3.1?**

This allows you to customize the model to better suit specific tasks and domains. But fine-tuning your model, you can improve the model's accuracy, relevance, efficiency. This made the fine-tuned model more effective for specialized task or applications for example customer service, content generation, sentiment analysis and others.

**What datasets are needed for fine-tuning?**

You’ll need a high-quality, well-structured dataset that includes diverse examples of the tasks the model will perform. These datasets can be manually curated, generated from existing logs, or a combination of both to ensure the model learns the specific nuances of your application.

**Best Known Method for Fine-tuning**
- **Start Simple** : Always beging your Fine-Tuning with a small datasets and mininal tarining paraemters like epochs.
- **Iterate Frequently** : Over the time, review and refine your dataset and model.

**Let's Get Started**

**1) Define Goals**
- It is important to define your fine-tuning objectives.
- Define the measure of success of your projects such as acheiving accuracy, faster response time or positive user feedback over a period of time.

**2) Prepare Your Dataset**
- A high-quality fine-tuning dataset is the backbone of successful fine-tuning. Your dataset should include a diverse set of examples that reflect the scenarios the model will encounter in production.
- Manual Data Collection: You can manually gather and edit examples. This is especially useful if you have domain-specific knowledge.
- Use Existing Logs: If you have an application already in production, you can use logs of real user interactions as training data.

**3) Select Base Model**
Selecting the right base model is important. Llama 3.1 comes in various sizes, each offering a trade-off between performance and resource requirements. At FinetuneDB we offer the following:

**Llama 3.1 70B**
Context: 128K, Speed: Medium
With a significantly larger parameter set, 70B excels in dealing with extensive datasets, producing highly sophisticated and contextually rich responses.

**Llama 3.1 8B**
Context: 128K, Speed: Fast
Has been refined to tackle complex tasks more efficiently than its predecessors. It has improved in handling multi-step tasks with better alignment and response diversity.

**4) Configure Training Parameters**
Below are few training parameters which you can adjust as needed during fine-tuning process

**Learning Rate**: The learning rate determines the size of the steps the model takes during training. A smaller learning rate may be useful to avoid overfitting.

**Batch Size**: That’s the number of examples in each batch. A larger batch size means that model parameters are updated less frequently, but with lower variance.

**Epochs**: An epoch refers to one full cycle through the training dataset. More epochs mean more training, but the risk of overfitting increases.

**5) Start Fine-Tuning**
Once everything has been set, we can initiate fine-tuning process. Depending on your dataset size, model complexity & hardware used (Xeon CPU, GPU or AI Accelerator Cards), fine tuning process can take anywhere from a few minutes to several hours.

**6) Evaluate Model Performance**
Monitoring metrics such as accuracy, loss, precision, and recall provide insights into the model's effectiveness and generalization capabilities. By assessing these metrics, you can gauge how well the fine-tuned model is performing on the task-specific data and identify potential areas for improvement.

**7) Deploy Your Model**
Once you are satisfied with your model's performance, can you deploy the model in production using several method for example:
- Hugging Face Inference Endpoints
- Amazon Sage Maker
- Azure Machine Learning
- FastAPI End Point
- vLLM
- TGI

**Let's focus on Step 2 to 5 for now**

**Supervised Fine-tuning (SFT): Train the model on your dataset using labeled examples where the desired outputs are provided. This is a common approach for tasks like text classification or question answering.**

1) The dataset which we will be using for this fine-tuning exercise will be "mlabonne/guanaco-llama2-1k"
2) We will be using Supervised Fine-Tuning Trainer from HuggingFace (https://huggingface.co/docs/trl/en/sft_trainer)
3) The base pretrained model that we will be using for this exercise will be from Meta Llama-3.1-8B (https://huggingface.co/meta-llama/Llama-3.1-8B)
 

**Complete instruction on how to run python script on running Meta-Llama-3.1-8B-Instruct with Intel Xeon 4th Gen and Newer with IPEX**

Steps to run this demo:

1) Clone this project into your local directory:
```  
   git clone https://github.com/allenwsh82/FT_llama_3.1.git
```
   
2) Create a new virtual environment inside the project which you just clone:
```  
   python -m venv lab_env 
```
   
3) Activate the virtual environment which you just created:
```
   source llm_chatbot_env/bin/activate 
```

4) Install the dependencies by running:
```
   pip install -r requirements.txt 
```

**Supervised Fine-Tuning**

5) Next, we are going to run the FT.sh file for the Supervised Fine Tuning using SFT Trainer Script from Hugging Face
```
   ./FT.sh
```
The total training time should take around 67 minutes   

![SFT](https://github.com/user-attachments/assets/0676fbca-b32a-4d88-8379-1b20b6f333a9)




5) If you go to the inference_bf16_ipex.py script, you will notice where two lines of code are added to enable AMX AI Accelerator to boost up performance:



**Let's test the fine-tuned model response with the following prompts:**
```
Please tell me whether Facebook, Instagram or Youtube is better! Thanks.
```
```
What is the difference between an ocean and a sea?
```
