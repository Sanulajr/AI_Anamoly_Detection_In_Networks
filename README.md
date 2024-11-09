# Network Anomaly Detection Project

## Project Overview

This project aims to detect anomalies in network traffic by using machine learning models to classify normal and malicious activities in a network. The dataset used is sourced from Kaggle's [Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection), providing a rich set of features for both the training and testing phases of the model. As part of my studies in Data Analytics and AI, I built this project to enhance my knowledge of machine learning techniques and network security.

## Key Features:
- **Anomaly Detection**: Detects abnormal network activities such as attacks or intrusions.
- **Model Comparison**: Utilized three machine learning models for comparison: 
  - **KNeighborsClassifier** 
  - **Logistic Regression**
  - **DecisionTreeClassifier**
  
- **Evaluation Metrics**: The models were evaluated based on precision and recall, with results providing insights into the trade-offs between false positives and false negatives.

## Methodology
1. **Data Preprocessing**: The dataset was preprocessed by handling missing values and normalizing features to ensure high-quality input for the models.
2. **Model Training and Optimization**: 
   - KNeighborsClassifier was optimized for accuracy in predicting network intrusions.
   - Logistic Regression provided a simpler yet effective model for anomaly detection.
   - DecisionTreeClassifier was tested for its interpretability and performance in classifying network anomalies.
   
3. **Model Comparison**: The models were evaluated using precision and recall metrics to assess their performance in real-world anomaly detection scenarios. The results were compared to select the best-performing model.

4. **Visualization**: A bar chart was generated to visualize the comparison of precision and recall scores for each model, offering a clear understanding of their relative performances.

## Results
- The **KNeighborsClassifier** achieved the highest precision and recall scores, indicating its strong performance in detecting both normal and abnormal network traffic.
- The **Logistic Regression** model was effective but showed lower performance in comparison to KNeighborsClassifier, particularly in recall.
- The **DecisionTreeClassifier** performed well but had a slightly lower precision, indicating potential overfitting for certain types of anomalies.

## Conclusion
This project was an excellent opportunity to apply machine learning techniques to a practical and impactful domain like network security. The results demonstrate that different models can have varying performances depending on the data and task at hand. As a student, this project allowed me to deepen my understanding of machine learning, data preprocessing, and anomaly detection, preparing me for real-world applications in the field of network security.

## Future Work
- **Model Enhancement**: Explore more advanced models like Random Forests or Neural Networks to further improve detection accuracy.
- **Real-Time Detection**: Implement the model for real-time anomaly detection in network systems.
- **Feature Engineering**: Investigate additional features to improve model performance.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Dataset
- Kaggle's [Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
