# Fake News Detection Project

## **Introduction**

In today's world, fake news spreads very fast through online platforms, social media, and websites. Fake news can mislead people and create confusion. This project aims to develop a **machine learning model** that can detect whether a news article is **real or fake**.

We have used **four different machine learning models** to classify news articles:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Gradient Boost Classifier**
- **Random Forest Classifier**

By comparing these models, we can find out which one gives the best accuracy in detecting fake news.

---
## **Project Overview**

Fake news is a big problem because it spreads false information and influences people’s opinions. This project uses machine learning algorithms to **analyze news articles and predict whether they are real or fake**. By using this model, we can help reduce the spread of misinformation.

### **How It Works**
1. We collect a dataset of news articles that are labeled as **True (Real News)** or **False (Fake News)**.
2. We clean and process the data so that our model can understand it better.
3. We train four different machine learning models and compare their results.
4. We evaluate each model’s performance using **accuracy, precision, recall, and F1 score**.
5. Finally, we use the best model to predict whether new articles are fake or real.

---
## **Dataset**

We used a labeled dataset that contains news articles along with their classification:
- **True (Real News):** News articles from reliable sources.
- **False (Fake News):** Fabricated or misleading news articles.

The dataset is divided into two parts:
- **Training Data:** Used to train the model.
- **Testing Data:** Used to check how well the model performs.

---
## **System Requirements**

### **Hardware Requirements**
- **Minimum 4GB RAM** (For smooth execution of the model)
- **Intel i3 Processor or higher**
- **500MB free disk space**

### **Software Requirements**
- **Python (Version 3.x)**
- **Anaconda (Optional, for Jupyter Notebook support)**
- **Code Editor (Jupyter Notebook, VS Code, or PyCharm)**

---
## **Dependencies**
Before running the project, you need to install the following Python libraries:

```sh
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn
pip install re
```

These libraries help in **data processing, visualization, and model training**.

---
## **How to Run the Project**

### **Step 1: Clone the Repository**
First, download the project files using Git:
```sh
git clone https://github.com/kapilsinghnegi/Fake-News-Detection.git
```

### **Step 2: Navigate to the Project Folder**
```sh
cd fake-news-detection
```

### **Step 3: Train and Test the Model**
Run the Python scripts for different classifiers. For example, to run the **Random Forest Classifier**, use:
```sh
python random_forest_classifier.py
```

The model will analyze news articles and predict if they are **real or fake**.

---
## **Results and Evaluation**

After training and testing the models, we evaluate their performance using:
- **Accuracy:** How often the model makes correct predictions.
- **Precision:** How many predicted fake news articles are actually fake.
- **Recall:** How many real fake news articles were correctly identified.
- **F1 Score:** A balance between precision and recall.

Each classifier gives different results. We compare them and choose the best one for fake news detection.

---
## **Conclusion**

This project successfully builds a **fake news detection system using machine learning**. By using different classifiers, we can find the best-performing model that accurately detects fake news. This system can be further improved with **more data and advanced deep learning models**.

Fake news is a growing problem, and this project is a step towards solving it with technology. 😊🚀

