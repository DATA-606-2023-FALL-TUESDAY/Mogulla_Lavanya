# Project Proposal

## 1. Project Title: Rice Grain Classification and Analysis for Quality Assurance.
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- **Author Name** - Lavanya Mogulla
- [Github]
- [LinkedIn] 
- **Powerpoint Presentation File** - In Progress
- **Youtube Video** - In Progress

[Github]: https://github.com/lavanyamogulla
[Linkedin]: https://www.linkedin.com/in/lavanya-mogulla/

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

**Data Visualization**

 - Utilized two popular plotting libraries, Matplotlib and Plotly Express, to display and inspect samples from each category in the dataset.

<img src="https://github.com/DATA-606-2023-FALL-TUESDAY/Mogulla_Lavanya/blob/main/data/newplot.png">

**Data Augmentation**

- This is a technique used to artificially increase the size of a dataset by creating modified versions of existing data.
- Enhanced the dataset using TensorFlow's image tools also applied several modifications to the original images, like sharpening, rotating, resizing, cropping. These changes can help the neural network generalize better when training.
