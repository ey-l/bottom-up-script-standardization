# How would you prompt it?

## Survey

We are using GPT as a baseline in our recent work, Automatically standardize data preparation scripts. However, we are concerned about potential inadequate prompt engineering, which may put the GPT models in disadvantage. Hence, we decided to conduct a survey to crowd-source prompt designs. 

Here is the setup:
* Suppose you have written a Python script to win a Kaggle competition, Titanic.
* You have access to a collection of Python scripts (https://www.kaggle.com/competitions/titanic/code) written by other competitors on the same dataset. You have the scripts downloaded locally. 
* You wonder if your script is missing some crucial steps that are present in the collection, or your script has anomaly steps that are not used by other competitors. 

Now, your goal is to use GPT and the script collection to improve your script. Please design a well-structured prompt to guide GPT to output a Python script.

## Responses

### Response 1

I want to preprocess a dataset from a Kaggle competition, and wrote the following Python script for that:
<user python script>
However, I'm unsure if I'm missing any steps or should leave some steps out. Specifically, I want my Python script above to be similar to what other people are doing. I gave you three scripts of other people below. Please alter my Python program above, and include or exclude steps based on what the other scripts below do. Your output should consist of one python script that compiles and runs, nothing more.
Other script 1:
<randomly chosen script 1>
Other script 2:
<randomly chosen script 2>
Other script 3:
<randomly chosen script 3>

### Response 2

I have written a Python script for a Kaggle competition:\n```python\n{code}\n```\n
Here are a set of script examples for the same competition:
```
[other scripts]
```
Assume you are an experienced data scientist. Please find out if my script is missing crucial steps or has unusual steps compared to the above script examples.
Several thoughts on making the script examples shorter:
1. Select several representative scripts
2. Remove unnecessary segments (e.g., comments, importing libraries)
3. Focus on one code module at a time

### Response 3

My goal is to train a machine learning model for the Kaggle competition, Titanic. The model aims to predict the chances of survival of a passenger on the Titanic using other traits. In the dataset, each training entry is a row, consisting of the following columns: passenger Id (int), survived (target prediction column, boolean), pclass(enum 1, 2, 3), name (string), sex (enum (male, female)), age (float), sibsp (boolean), parch (boolean), ticket (string), fare (float), cabin (string), embarked (enum(C, S)). Using the information above, please check this data pre-processing script. \n```python\n{code}\n```\n In particular, what are the steps that looks abnormal compared to other people's scripts below? What are some steps that are generally included in other people's scripts that are not in mine? \n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\n```python\n{scripts_sampled[3]}\n```\n```python\n{scripts_sampled[4]}\n```\n

### Response 4

I'm working on kaggle Titanic competition: https://www.kaggle.com/competitions/titanic/code. I'm not sure about whether my script is correct, i.e., my script may miss some crucial steps that are present in the collection, or my script has anomaly steps that are not used by other competitors. Can you check the following script:\n```python\n{code}\n```\nCan you directly optimize my script? \nNow output a Python script. You should only return straight-line Python code.

### Response 5

Imagine you are trying to write a python script for a data analysis job. You are given a csv file that include passenger information of the Titanic like name, age, gender, socio-economic class, etc., and whether they survived or not. You want to use this csv to train a machine learning model to predict whether someone is likely to survive or not. Write a python script that does this. Here is a starting point: \n```python\n{code}\n```\n


### Response 6

Here is some a better description of the csv: Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)
parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.. See if you can improve the preprocessing step by making it more standardized and robust:
[insert script]

### Response 7

f"Given a collection of Python scripts from the Kaggle Titanic competition \n```python\n{scripts_sampled[0]}\n```\n```python\n{scripts_sampled[1]}\n```\n\n```python\n{scripts_sampled[2]}\n```\n```python\n{scripts_sampled[3]}\n```\n```python\n{scripts_sampled[4]}\n```\n, I am looking to improve my current script. Here is my current script:\n```python\n{code}\n```\nPlease analyze the collection of scripts and suggest improvements or additional steps that could potentially enhance the performance of my script. The improvements could be in terms of model accuracy or any other aspect that you find relevant. Please provide the improved Python script.\nNow output a Python script. You should only return straight-line Python code."

### Response 8

Read the following collection of python scripts (https://www.kaggle.com/competitions/titanic/code.) from a Kaggle contest on Titanic dataset and extract common patterns and best practices in them. Input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code.

### Response 9

Now I will give you another script for the same contest. \n Your job is to improve this script in terms of <runtime performance|accuracy|recall> using the common patterns and best practices you have learned from the collection.\n You are allowed to add and delete steps.  \n Step is defined as any valid python statement.   \n Make sure your output is a syntactically correct python script.\n Here is the input: \n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code.

### Response 10

—\n I am working on a Kaggle competition using the titanic dataset. I want you help me with the preprocessing of the dataset, using standard winning methods.\n For example, automatically add things like:\n - Imputation\n - Normalization.\n - Anything else that might help.\n Also remove bad, useless, or "illegal" steps.\n "Illegal" steps are things like including the target (survived or not) in the final output. Likewise faking a column strongly correlated (like survived + 1). I want you to remove these bad steps, and other pointless steps.\n —\nHere are a few tips:\n- Assume there is a local file called "titanic.csv" containing the training data.\n- Use pandas to read it.\n- Use anything you see fit for preprocessing.\n—\nI will give you as input a partially written script (potentially empty). Your output should be a complete preprocessing script, including all the imports (e.g., using pandas). Format your output as follows:(START)\n```python```\n(END)\n—\nHere is the input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code.

### Response 11

Pretend you are an expert data analyst with extensive knowledge and expertise in data analysis, data cleaning, data wrangling, and machine learning. Write a Python program to achieve high accuracy on the Titanic - Machine Learning from Disaster challenge on Kaggle. You have analyzed the wealth of information that is available online for solving this challenge, such as code repositories, blog articles, tutorials, etc. List a few advanced machine learning models that others have used for this challenge. What is the reported accuracy of these methods? Use the model with the highest accuracy for the code. Similarly, what data cleaning and wrangling methods are used for this challenge? Use the techniques that are widely used by others to achieve high accuracy on this challenge in the Python code. Here is the input:\n```python\n{code}\n```\n\nNow output a Python script. You should only return straight-line Python code.

### Response 12

"I have a Python script that I would like you to edit. " \
    + 'The modified script should be able to run and maintain similar semantics to that of the unmodified version.' \
    + f"The script is from Kaggle's '{competition_full_name}' competition." \
    + "The goal is to make the script more standard regarding other scripts in the same competition. That means it should use common libraries, common functions, etc." \
    + "You should only return the code." \
    + "I am sending it as an attachment now. \n```python \n"

