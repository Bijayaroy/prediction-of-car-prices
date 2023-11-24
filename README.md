# Predictive Modeling for Car Prices

## Introduction :
The objective of this project was to build predictive models for car prices using various machine-learning algorithms. 
The dataset consisted of multiple features, including company, name (model name), year, selling_price, km_driven, fuel, seller_type, transmission, and owner.
Machine Learning has gained significant momentum in the past decade.
It would be highly advantageous to forecast car prices based on a set of features like company, fuel, and other relevant factors.
Consider a scenario where a company aims to determine the price of a car based on features such as year, company, and transmission, and other features. Employing machine learning models can aid in accurately setting car prices, and maximizing profits for the company.
Utilizing machine learning models ensures precision in establishing optimal prices for new cars, leading to substantial cost savings for car manufacturers.
Our focus will be on working with car price prediction data and generating forecasts for various types of cars.
Our initial steps involve data visualization to gain insights and comprehend crucial information essential for accurate predictions.
Various regression techniques will be applied to derive the average price of the considered car, contributing to informed decision-making in pricing strategies.

## Data Exploration :
In this phase, we delve into the dataset to understand its structure and the information it holds. Our dataset consists of nine columns, each providing different details about cars.
By conducting this exploration, we aim to gain insights into the dataset's characteristics and identify potential patterns or trends that could influence car prices. This initial understanding will guide our subsequent steps in data preprocessing, visualization, and modeling.

The dataset that has been used is taken from Kaggle:
https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset

I have added a column 'company' to aid the use of the dataset.

## Data Visualization :
In this phase, we leverage visualizations to gain insights into the distribution of the target variable (selling_price) and explore relationships between this target variable and other features present in the dataset. Through these visualizations, we aim to uncover patterns, trends, and potential outliers in the data, which will inform subsequent steps in feature selection and model building. The visual insights obtained will contribute to a more nuanced understanding of the factors influencing car prices in our dataset.

## Data Preprocessing :
In the data preprocessing phase, our primary focus is on preparing the dataset for effective analysis and modeling. This involves addressing missing values, handling outliers, and converting categorical variables into a format suitable for machine learning algorithms.

## Feature Selection :
Selecting the right features is crucial for building an accurate and efficient predictive model for car prices. We employ feature selection techniques to identify and justify the inclusion of relevant features in our analysis. One common method for this purpose is correlation analysis, which helps us understand the relationships between different features and the target variable (selling price).
Correlation analysis measures the strength and direction of the linear relationship between two variables. In our context, we are interested in the correlation between each feature and the selling price of cars. 
Heatmap is useful under the Seaborn library. It gives us a good colored estimate of the values. Depending on the palette chosen, we either get bright images for higher values or vice-versa. We are plotting a correlation plot between the various features.
It is seen that the ‘selling_price’ and ‘transmission’ are better correlated to each other if not the best, with a correlation coefficient being equal to 0.53. Similarly, the features ‘selling_price’ and ‘year’ are also related. All the remaining features seem to be related in a negative way or uncorrelated.
The values of the correlation coefficient lie between the range -1 to 1 respectively. The higher the positive correlation between the features, the more would be correlation coefficient value would move to 1. The higher the negative correlation between the features, the more would the correlation coefficient value move to -1 respectively.

Justification : Our goal in selecting features is to build a model that is both accurate and interpretable. By choosing features with strong correlations with the selling price, we increase the model's predictive power. Additionally, considering domain knowledge ensures that we capture relevant factors that might not be fully reflected in the correlation coefficients.
Through this process, we strike a balance between statistical evidence and real-world understanding, ultimately selecting a subset of features that contribute meaningfully to predicting car prices. This thoughtful feature selection enhances the model's performance and interpretability.

## Model Building :
In this phase, we embark on constructing a predictive model for car prices based on our dataset. The process involves splitting the data into training and testing sets to ensure the model's effectiveness and selecting a suitable regression model, such as linear regression, for training.
Dataset Splitting:
The dataset is divided into two subsets - a training set and a testing set. The training set, typically comprising 80% of the data, serves as the foundation for teaching the model. The remaining 20%, allocated to the testing set, acts as a completely independent dataset to assess the model's performance.
Purpose:
By splitting the dataset, we simulate a real-world scenario where the model encounters new, unseen data. This evaluation on an untouched dataset ensures the model's ability to generalize to different instances beyond the ones it learned during training.

Choosing a Regression Model:
Selection of Linear Regression: Linear regression is a fundamental and interpretable regression model that assumes a linear relationship between the features and the target variable. For predicting car prices, it's a logical starting point. The model assumes that changes in the target variable are proportional to changes in the input features, making it a suitable choice for our analysis.
Training the Model:
The selected linear regression model is trained on the training set. During training, the model learns the relationships between the features (such as year, fuel, etc.) and the target variable (selling_price) by adjusting its parameters.

All the Models that are	used:
1.	Linear Regression
2.	Support Vector Regressor	
3.	K Nearest Regressor	
4.	Decision Tree Regressor	
5.	Gradient Boosting Regressor	
6.	MLP Regressor	

## Model Evaluation :
In evaluating the performance of various machine learning models on a given dataset, two commonly used metrics are the Mean Absolute Error (MAE) and Mean Squared Error (MSE). These metrics help quantify the accuracy and precision of a model's predictions. Let's analyze the performance of the different models based on the provided MAE and MSE values.
Here is a summary of the performance metrics for each model:
Linear Regression Model:
MAE: 182,151
MSE: 95,921,662,170.
Support Vector Regressor:
MAE: 301,113
MSE: 309,076,488,744.
K Nearest Regressor:
MAE: 115,718
MSE: 67,248,191,409.
Decision Tree Regressor:
MAE: 120,292
MSE: 82,017,401,002.
Gradient Boosting Regressor:
MAE: 132,611
MSE: 91,443,743,824.
MLP Regressor:
MAE: 145,457
MSE: 97,817,839,711.

Let's discuss the strengths and weaknesses of the K Nearest Neighbors (KNN) Regressor based on its performance metrics and general characteristics:

Strengths:
Simple and Intuitive: KNN is a simple algorithm that is easy to understand and implement. It doesn't make strong assumptions about the underlying data distribution.
No Training Phase: KNN is a lazy learner, meaning it doesn't have a training phase. This can be advantageous when dealing with dynamic datasets where the relationships between variables may change over time.
Effective for Local Patterns: KNN is particularly effective when the underlying data has local patterns or when instances with similar feature values tend to have similar target values.
Non-Parametric: KNN is a non-parametric algorithm, meaning it doesn't assume a specific form for the underlying data distribution. This flexibility allows it to perform well on diverse types of datasets.

Weaknesses:
Computational Cost: The main drawback of KNN is its computational cost during prediction, especially on large datasets. Calculating distances between the query point and all data points can be time-consuming.
Sensitive to Outliers: KNN is sensitive to outliers in the dataset. Outliers can significantly impact the distance calculations and, consequently, the predictions.
Curse of Dimensionality: KNN's performance may degrade as the number of features or dimensions increases. This is known as the curse of dimensionality, where the distance between points becomes less meaningful in high-dimensional spaces.
Need for Optimal K: The choice of the number of neighbors (K) can influence the model's performance. Too small a value may lead to overfitting, while too large a value may result in underfitting. Optimal K selection often requires experimentation.
Imbalanced Data: In datasets with imbalanced class distributions, KNN may favor the majority class, leading to suboptimal predictions for the minority class.

## Reflection and Improvement :
Let's delve into a technical analysis of each of the models used in the car price prediction project:
1. Linear Regression Model:
Strengths:
Simplicity: Linear regression is straightforward and easy to interpret.
Fast Training: The model trains quickly on large datasets.
Weaknesses:
Limited Complexity: Might struggle to capture complex relationships in the data.
Assumes Linearity: Assumes a linear relationship between features and target, which may not hold in all cases.

2. Support Vector Regressor (SVR):
Strengths:
Effective in High-Dimensional Spaces: Performs well when the number of features is high.
Robust to Overfitting: Reduces the risk of overfitting in high-dimensional spaces.
Weaknesses:
Computationally Intensive: Can be slow on large datasets due to complex computations.
Requires Tuning: Performance depends on choosing appropriate kernel and hyperparameters.

3. K Nearest Regressor (KNN):
Strengths:
Local Patterns: Effective for capturing local patterns and relationships in the data.
Non-Parametric: No assumptions about the underlying data distribution.
Weaknesses:
Computational Cost: High computational cost during prediction, especially on large datasets.
Sensitive to Outliers: Outliers can significantly impact predictions.

4. Decision Tree Regressor:
Strengths:
Non-Linear Relationships: Can capture non-linear relationships in the data.
Interpretable: Easy to interpret and visualize.
Weaknesses:
Overfitting: Prone to overfitting, especially on small datasets.
Lack of Smoothness: Decision trees can create fragmented, less smooth decision boundaries.

5. Gradient Boosting Regressor:
Strengths:
Ensemble Learning: Combines multiple weak learners for a strong model.
Handles Non-Linearity: Can capture complex relationships in the data.
Weaknesses:
Computationally Intensive: Training can be time-consuming and resource-intensive.
Requires Tuning: Performance depends on tuning parameters like learning rate and tree depth.

6. MLP Regressor (Multi-Layer Perceptron):
Strengths:
Neural Network Power: Can model complex relationships through multiple layers.
Non-Linear Activation: Uses non-linear activation functions for flexibility.
Weaknesses:
Training Complexity: Training deep networks can be complex and time-consuming.
Sensitive to Initialization: Performance can depend on the initial weights of the network.

Overall Analysis:
K Nearest Regressor: Performs well with low MAE, indicating accurate predictions. However, computational cost is a consideration.
Decision Tree Regressor: Shows promise with a relatively low MAE. Prone to overfitting but interpretable.
Linear Regression: Simple but may not capture complex relationships well.
SVR, Gradient Boosting, MLP: Show decent performance but may require careful tuning and consideration of computational resources.

Recommendations:
K Nearest and Decision Tree: Recommended for accurate predictions.
Further Analysis: Explore tuning and cross-validation for all models to improve performance.

## Conclusion:
Considering both MAE and MSE, the K Nearest Regressor appears to be the best-performing model for this dataset, closely followed by the Decision Tree Regressor. These models demonstrate a better ability to make accurate predictions with lower errors compared to the other algorithms. It's essential to note that the choice of the best model may depend on specific requirements, such as interpretability, computational efficiency, and scalability. Further analysis and experimentation could be conducted, such as cross-validation or hyperparameter tuning, to ensure robust model selection.

## For Non-Technical Audience in brief :
Prediction of Car Prices:
1. Introduction:
We aimed to predict car prices using fancy math and machine learning! Our data had info like the car's year, transmission, and more.

2. Data Check:
We looked at the data, fixed missing stuff, and handled weird numbers. We turned words like "petrol" into numbers the computer understands.

3. Models and Numbers:
We tried different models to guess car prices. One model (K Nearest) did the best – it was closest to the real prices. Another good one was the Decision Tree model. Some models were okay, but one (Linear Regression) didn't catch tricky stuff in the data.

4. What We Found:
The K Nearest model was the star, making the smallest mistakes in guessing. Decision Tree was good too, not far behind. Linear Regression was decent but missed some details.

5. Conclusion:
So, K Nearest and Decision Tree are our heroes for predicting car prices. But, it depends on what you need – some models are good for different things. We might tweak them more to be even better.

6. What to Do Next:
Trust K Nearest and Decision Tree for car price guesses. We're not done yet – we'll check more things to make our guesses even sharper. Stay tuned for better car price predictions!
