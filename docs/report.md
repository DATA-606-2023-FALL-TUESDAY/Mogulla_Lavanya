# Project Report

## Lavanya Mogulla, 3rd Semester

## 1. Project Title: Rice Grain Classification and Analysis for Quality Assurance.
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [Github]
- [LinkedIn] 
- [Powerpoint]
- [Youtube Video]

[Github]: https://github.com/lavanyamogulla
[Linkedin]: https://www.linkedin.com/in/lavanya-mogulla/
[Powerpoint]: https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/src/Presentation.pptx
[Youtube Video]: https://www.youtube.com/watch?v=4uPEvulaA3Q

 ## 2. Background
Image processing and computer vision applications in agriculture are of interest due to their non-destructive evaluation and low cost compared to manual methods. Rice is a globally produced grain with numerous genetic varieties, distinguished by traits like texture, shape, and color. A study in Turkey examined Arborio, Basmati, Ipsala, Jasmine, and Karacadag rice varieties to classify and assess seed quality based on these distinguishing features, which are essential for both agriculture and culinary purposes. ANN and DNN are used to model feature datasets, while a CNN is used for image datasets. Classification will be conducted, and statistical metrics including sensitivity, specificity, prediction, F1 score, accuracy, false positive rate, and false negative rate will be computed from the confusion matrix values. 

The model also assesses rice grain quality by analyzing size, color, and condition, including broken or dusted grains. It offers automated detection to optimize production, minimize waste, and ensure consistent product quality. It provides detailed reports and statistics for quality assurance and compliance.

**Research Questions**
- What are the most critical features in the feature dataset for accurately classifying rice grain quality, and how do these features correlate with visual characteristics observed in the image dataset?
- How can the model contribute to sustainable rice production by reducing resource wastage and optimizing production processes?
- What kind of detailed reports and statistics on rice quality can the model provide, and how can these be effectively utilized for quality assurance and compliance with industry standards?

 ## 3. Data

  - **Data Source** - This dataset is collected from Kaggle.

     **Dataset Link** - [Rice Data]
- **Data size** -  267.35 MB
- **Number of Images** - 75k
- **Number of Classes** - 5
- **Class Labels** - Arborio, Basmati, Ipsala, Jasmine and Karacadag
  
   Each class has equal data of 15k images.


  **Samples of Rice**

<img src="https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Rice%20sample.png">

   
[Rice Data]: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset


## 4. Exploratory Data Analysis (EDA)

**Data Preprocessing**

- Due to limitations in GPU resources, a decision was made to utilize a dataset comprising 6,000 images for each class. 
- This approach was adopted to ensure efficient processing and analysis within the available computational constraints.
- In extending the scope of this project beyond mere rice classification, we've incorporated a crucial element of quality assessment to offer a more comprehensive and practical analysis of rice varieties.

<img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Data.png">

**Data Visualization**

 - Utilized two popular plotting libraries, Matplotlib and Plotly Express, to display and inspect samples from each category in the dataset.

<img src="https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/newplot.png">

**Value Counts for each type of Rice after Quality assessment**

<img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Value%20counts%20for%20each%20type%20of%20rice.png">

## 5. Model Training

**Utilization of Google Colab Pro for Model Training**

- **High-Performance Computing:** Leveraged Google Colab Pro for enhanced RAM and GPU capabilities, crucial for efficient model training.
- **Cost-Effectiveness:** $10.5 per month for every 100 compute units, providing a balance between computational power and budget.
- **Model Training Benefits:** Accelerated training times due to high RAM and GPU capacity. Enabled handling of large datasets and complex computations.
- **Platform Accessibility:** Easy access to advanced computing resources without the need for personal hardware upgrades.

**Tensor Flow Image Classification Pipeline**

This is a series of steps utilized to develop a model capable of categorizing images into predefined classes. This pipeline typically involves the following stages:

- **Data Preparation :** 
  The dataset is partitioned into training and testing subsets with a 33% test size allocation, and image labels were numerically encoded and assigned across both sets.

- **Image Loading and Pre-processing :**
  A custom load_image function is implemented to read and decode JPEG images, complemented by a preprocess_image function that resizes, normalizes images, and 
  applies one-hot encoding to the labels.

- **Data Augumentation :**
  This is a technique used to artificially increase the size of a dataset by creating modified versions of existing data. Enhanced the dataset using TensorFlow's image tools, also applied several modifications to the original images, like sharpening, rotating, resizing, cropping. These changes can help the neural network generalize better when training.

- **Tensor Flow Data Generators :**
  The tfdata_generator function is used to create efficient data pipelines. It utilizes caching, shuffling, batching, and prefetching for optimal performance and conditional augmentation during training phase.

- **Dataset Handling :**
 The batch size is set to 32 for manageable memory usage and generators are separated for training and testing datasets.

**Model Architecture**

<img src = "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Architecture.png">

**InceptionV3 Model**

 - **Input Layer:** 
This layer receives images with shape (400, 400, 3).
- **InceptionV3 Base:**
It is pre-trained on ImageNet, extracts complex features from input images and all layers set to trainable, allowing model adaptation to new data.
- **Conv2D Layers :**
Then apply convolution operations to extract high-level features and the Kernel regularizer (L2) helps in reducing overfitting.
- **MaxPooling2D Layers:**
This layer reduces spatial dimensions (size of the feature maps), focusing on important features.
- **GlobalAveragePooling2D:**
  This layer reduces each feature map to a single average value, decreasing model complexity and computations.
- **Flatten Layer:**
It converts pooled feature maps into a single vector, preparing for fully connected layers.
- **Dense Layers:**
  These fully connected layers learn non-linear combinations of high-level features and apply L2 regularization and dropout to prevent overfitting.
- **Output Layer (Dense):**
This is the final layer with softmax activation for multi-class classification.

<img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Model.png">

- In addition to tensorflow, pandas, numpy, keras, plotly are used for this project.

  **Pandas:** Pandas is a powerful data manipulation and analysis library for Python, providing extensive capabilities for data preparation, cleaning, and exploration with its DataFrame object.

  **NumPy:** NumPy is the fundamental package for scientific computing in Python, offering support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays.

  **Plotly:** Plotly is an interactive graphing library for Python that enables users to create visually appealing and sophisticated plots and charts that can be displayed in a web browser.

  **Keras:** Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano, designed for enabling fast experimentation with deep neural networks.

- **Model Comparision :** The InceptionV3 model was selected for this dataset because of its history of providing high accuracy. It proved to be potentially better model than the others. In order to guarantee a comprehensive assessment of the possibilities available, efforts were made to compare the InceptionV3 against a different model. Regretfully, insufficient GPU resources prevented the comparison study, making it unable to finish this parallel assessment. This constraint emphasizes how important it is to have strong computing power for modern machine learning projects. Access to more potent GPUs would facilitate future research in thoroughly examining and verifying the relative benefits of different models.

- **Accuracy v/s Epoch Plot**

  <img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Accuracy.png">

- **Loss v/s Epoch Plot**

  <img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Loss.png">

  ## 6. Application of Trained Model

  **Flask**  - Flask, a lightweight Python web framework, offers an ideal environment for deploying trained machine learning models into web applications. It allows for the easy creation of RESTful APIs, which can process user inputs through the model and return predictions. Flask excels in handling HTTP requests and responses, enabling seamless interactions between the user and the model. Its support for template rendering makes it straightforward to display complex model outputs in a user-friendly format. Additionally, Flask's extensive community support and range of extensions provide added flexibility and functionality, making it a versatile choice for integrating machine learning models into web-based platforms. This framework bridges the gap between data science and web development, allowing for the creation of dynamic, interactive applications that leverage the power of machine learning.

  **1. app.py:** This is the main Python file where your Flask application is defined and configured. It includes route definitions, which are functions that specify what the server should do when a URL is accessed.

  **2. demo.html:** This is an HTML template that displays the results of the model's predictions. When a user submits data through the application, demo.html is rendered with the output.
  
  **3. home.html:** This HTML file serves as the landing page of your Flask application. It generally contains the user interface for inputting data into the model. It's designed to be user-friendly and intuitive, guiding users on how to interact with the application.

<img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Home%20Page.png">

<img src= "https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/Demo%20Page.png">

## 7. Conclusion

**Summary**

This project marks a significant advancement in agri-tech by leveraging machine learning to accurately classify rice varieties and assess quality. This innovation offers a scalable alternative to manual inspection, enhancing efficiency and precision in agricultural practices and food quality control. The project's success paves the way for future technological integration into the agricultural sector, promising improved standards and food security.

**Limitations**

- Rice comes in numerous varieties, each with its unique characteristics. The model might struggle to accurately classify or assess quality if it hasn't been trained on a sufficiently diverse range of rice types.
  
- High-resolution image processing and complex algorithmic analysis require significant computational resources, which might not be feasible in resource-limited settings.

- Performing real-time quality analysis in a field setting poses technical challenges, including the need for mobile compatibility and offline functionality.

**Future Research**

Expanding the dataset to include a wider variety of rice types from different regions and under varied conditions to improve the model's robustness and accuracy. Also, researching ways to integrate the model with IoT devices for automated data collection and analysis, which could include drones or automated field cameras. Creating a user-friendly mobile application to make the technology more accessible to farmers and agricultural workers, especially in remote areas.  

## 8. References

1. Guan, Q., Wan, X., Lu, H., Ping, B., Li, D., Wang, L., Zhu, Y., Wang, Y., &amp; Xiang, J. (2019, July). Deep Convolutional Neural Network inception-V3 model for differential diagnosing of lymph node in cytological images: A pilot study. Annals of translational medicine. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6694266/
   
2. Perera, L. (2023, January 8). How to create a simple API from a machine learning model in python using flask. Medium. https://lakshitha1629.medium.com/how-to-create-a-simple-api-from-a-machine-learning-model-in-python-using-flask-661e9d9c7633 







  



