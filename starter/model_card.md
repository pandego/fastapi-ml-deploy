# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Name**: Logistic Regression Classifier for Income Prediction
- **Version**: 1.0
- **Date**: October 11, 2023
- **Authors**: Miguel D.
- **License**: N/A

## Intended Use
This model is intended to predict if a person makes more than 50K a year based on various features like age, work class, education, marital status, etc. The primary use case is for automated income categorization in financial or demographic studies. 

## Training Data
- **Source**: U.S. Census Data
- **Features**:
  - Age
  - Workclass
  - FNLGT (Final Weight)
  - Education
  - Education Number
  - Marital Status
  - Occupation
  - Relationship
  - Race
  - Sex
  - Capital Gain
  - Capital Loss
  - Hours per Week
  - Native Country
- **Label**: Salary (Binary: `>50K`, `<=50K`)

## Evaluation Data
The model was evaluated using a randomly sampled test set comprising 20% of the original U.S. Census dataset.

## Metrics
- **Precision**: 0.7450
- **Recall**: 0.2608
- **F1 Score**: 0.3863

Note: The metrics indicate that while the model has a high precision (when it predicts `>50K`, it's correct 74.5% of the time), its recall is quite low (it identifies only 26.08% of all actual `>50K` instances). The F1 Score, which is the harmonic mean of precision and recall, is 0.3863, suggesting there's room for improvement in balancing precision and recall.

## Ethical Considerations
- **Bias and Fairness**: The model is trained on U.S. Census data, which might have inherent biases. It's essential to evaluate the model's fairness across different demographic groups.
- **Privacy**: Ensure that any data used with this model doesn't violate user privacy or data protection regulations.

## Caveats and Recommendations
- The model might not perform well for non-U.S. populations given the data it was trained on.
- Always consider the ethical implications when deploying in real-world scenarios, especially in applications that might impact an individual's financial opportunities.
- Further tuning or more complex models might be needed to improve recall without significantly sacrificing precision.
