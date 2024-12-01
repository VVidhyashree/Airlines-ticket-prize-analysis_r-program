# Airlines-ticket-prize-analysis_r-program
Analysis of airline ticket price analysis using r programming

Objective:
The objective of this project is to predict flight-related outcomes, including flight class (Economy or Business) and ticket price, based on various flight features such as airline, source city, departure time, number of stops, flight duration, and days left before departure. The project applies both classification (for predicting flight class) and regression (for predicting ticket price) tasks using machine learning algorithms.

Dataset:
The dataset contains flight-related data with the following key features:

Airline: The airline operating the flight.
Source City: The city from which the flight departs.
Departure Time: The scheduled time of departure.
Stops: The number of stops during the flight.
Duration: The duration of the flight.
Days Left: The number of days remaining before the flight departs.
Class: The target variable for the classification task, indicating whether the flight is in Economy or Business class.
Price: The target variable for the regression task, representing the price of the flight ticket.
Methodology:
Data Preprocessing:

Clean the dataset by handling missing values (if any).
Encode categorical variables (e.g., airline, source city) using appropriate techniques like one-hot encoding.
Normalize continuous variables (e.g., duration, days left) to bring them on the same scale.
Exploratory Data Analysis (EDA):

Visualize distributions of key features and relationships between them.
Analyze correlations between features and target variables to understand potential patterns.
Classification Task (Predict Flight Class):

Target Variable: Class (Economy vs. Business).
Model: Logistic Regression, a classification model that predicts the likelihood of a flight being in Business class based on the given features.
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Regression Task (Predict Ticket Price):

Target Variable: Price (Continuous numeric value).
Model: Gradient Boosting Regressor, a powerful regression algorithm used to predict the ticket price based on the given features.
Evaluation Metrics: Mean Squared Error (MSE), R-squared, and Root Mean Squared Error (RMSE).
Model Tuning:

Fine-tune the models by adjusting hyperparameters and evaluating performance through cross-validation.
Handle overfitting and underfitting by adjusting model complexity.
Deployment (Optional):

Deploy the trained models into a web application or API for real-time predictions.
Results:
For Classification (Flight Class): The logistic regression model predicts whether a flight will be in Business or Economy class based on input features. The model’s performance is evaluated using metrics like accuracy and confusion matrix.

For Regression (Ticket Price): The gradient boosting model predicts the ticket price of a flight. The model is evaluated using MSE and R-squared, providing insight into how well the model fits the data and predicts the price.

Insights & Conclusion:
This project demonstrates the power of machine learning techniques in solving real-world problems in the airline industry, such as predicting flight class and ticket price.
The classification model helps airlines make decisions on seat allocation and pricing strategies, while the regression model assists in estimating ticket prices based on various factors.
Future improvements could include incorporating additional features, experimenting with other machine learning algorithms (e.g., Random Forest, XGBoost), or deploying the models for real-time use.
