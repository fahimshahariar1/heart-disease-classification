# heart-disease-classification

This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable to predicting whether or not someone has heart disease based on their medical attributes.

We are going to take the following approach
1. Problem Definition
2. Data
3. Evaluation
4. Features
5. Modelling
6. Experimentation

## 1. Problem Definition: 
In a statement, Given clinical parameters about a patient, can we predict whether or not they have heart disease?

## 2. Data: 
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

## 3. Evaluation:
If we can reach 95% accuracy in predicting whether or not a patient has heart disease during the proof of concept, we will pursue the project

## Features:

1. age
2. sex
3. chest pain type (4 values)
4. resting blood pressure
5. serum cholestoral in mg/dl
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved
9. exercise induced angina
10. oldpeak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment
12. number of major vessels (0-3) colored by flourosopy
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
14. The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

### Load Data
Now we will load the data from the resources using df = pd.read_csv("heart-disease.csv")

### Data Exploration (Exploratory Data Analysis or EDA): 
The goal here is to find our more about the data and become a subject matter expert on the data set that we are working with
During exploration, we check if we have any missing values or not. what type of values we have for the target variable etc. to get familiarity about the data
df["target"].value_counts()
df["target"].value_counts().plot(kind = "bar", color=["salmon", 'lightblue']);
df.isna().sum()

### Heart Disease frequency based on sex
df.sex.value_counts()
we can compare it to crosstab using a bar plot
pd.crosstab(df.target, df.sex).plot(kind = "bar", color=["salmon", "lightblue"])
plt.title("Heart Disease based on sex")
plt.xlabel("0 = No, 1 = Yes")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation= 0);

As we are proceeding with the dataset. Our first goal would be to find out patterns inside our dataset that will help us use different algorithms to reach our target

### Age vs Max heart rate for heart Disease
plt.figure(figsize =(10,6))

plt.scatter(df.age[df.target == 1],
           df.thalach[df.target == 1],
           c = "salmon")

plt.scatter(df.age[df.target == 0],
           df.thalach[df.target == 0],
           c= "blue")

plt.title("Heart Disease in function of Age and Maximum heart rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", 'No Disease']);

### Heart Disease frequency based on chest pain type
cp: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic

Next up we will try to find the correlation matrix of the dataset 
#Correlation matrix using seaborn
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix, annot = True, linewidths=0.5, fmt='.2f', cmap='YlGnBu')

By this correlation matrix we are basically trying to find out patterns by which the target column is related or gets triggered. and we can find that lot of other columns can have an impact in the target column(e.g. CP, thalach etc)

## 5. Modelling
Now we will split the data into X and y first, then prepare them for training and testing sets so that we can run our model on them.

To do that we will first split the data into X & y
X = df.drop("target", axis =1)
y = df["target"]

Then prepare them or split them into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

No we have our data split into training and test sets. It's time to build a machine learning model.
We will train it find the patterns on the training set
then we will test it use the patterns on test set

We are going to use 3 models to do the prediction
1. Logistic Regression
2. K-nearest Neighbour
3. Random Forest Clasifier

Again, we find all these models by using the sklearn model map that you can find here https://scikit-learn.org/stable/machine_learning_map.html
But why are we using logistic regression if this is a classification problem? Well, that's the interesting part.
If you search for logistic regression, you will find in the documentation that it says, "Despite its name, it is implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature."
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

So these are the 3 models that we are going to use. Now we can't just create train and test sets and try to predict it for every model. Instead, we will create a dictionary and put all the models in it, and create a function to fit and score with those models

#Put models in a dictionary
models = {"Logistic Regression" : LogisticRegression(),
         "KNN" : KNeighborsClassifier(),
         "Random Forest": RandomForestRegressor()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models: a dict of different scikit-learn machine learning models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    y_train: training labels
    y_test: testing labels
    """
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
model_scores = fit_and_score(models=models, X_train=X_train, X_test = X_test, y_train = y_train, y_test = y_test)
model_scores

And we will see that the logistic regression scores the highest with 88% accuracy. But we are still not close enough to our target. Our target was to get 95% accuracy

Now that we have some default values, we need to do some tuning in order to get better accuracy.
Things that we are going to look into are
1. Hyperparameter tuning
2. Feature importance
3. Confusion matrix
4. Cross-validation
5. Precision
6. Recall
7. F1 Score
8. Classification Report
9. ROC Curve
10. Area under curve (AUC)

The first thing that we are going to tune is the Hyperparameter
### Hyperparameter Tuning
That means that we are going to tune few parameters of the algorithm to get the best possible value.
We will first start with KNN algorithm
#let's tune KNN
train_scores = []
test_scores = []

#create a list of different values for n-neighbors
neighbors = range(1,21)

#setup KNN
knn = KNeighborsClassifier()

#loop through different neighbors
for i in neighbors:
    knn.set_params(n_neighbors = i)
    #fit the algorithm
    knn.fit(X_train, y_train)
    #update the training score list
    train_scores.append(knn.score(X_train, y_train))
    #update the test score list
    test_scores.append(knn.score(X_test, y_test))

even after tuning we got the highest accuracy to be 67.21% which is very low compared to other two models. So for now we will discard this model and work with other two to get our desired accuracy
