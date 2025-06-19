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
