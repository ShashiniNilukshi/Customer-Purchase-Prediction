import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data =pd.read_excel('Example.xlsx')
print(data)

data.info()

# data.hist(bins=50,figsize=(18,8))
# plt.show()

data["Gender"].value_counts()
data["Marital_Status"].value_counts()
data["Education_Level"].value_counts()
data["Customer_Segment"].value_counts()
data["Purchase"].value_counts()

lable_encoder=LabelEncoder()
data["Gender"]=lable_encoder.fit_transform(data["Gender"])
data["Marital_Status"]=lable_encoder.fit_transform(data["Marital_Status"])
data["Education_Level"]=lable_encoder.fit_transform(data["Education_Level"])
data["Customer_Segment"]=lable_encoder.fit_transform(data["Customer_Segment"])


x=data[['Age','Gender','Income','Marital_Status','Education_Level','Previous_Purchases','Customer_Segment']]
y=data['Purchase']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Random_state is used to control the randomness of the data split. It ensures that the data is split the same way every time you run the code, which is helpful for reproducibility.

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()

print("Confusion Matrix:")
print(cm)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

with open('Customer_Purchase_Prediction_Model.pkl','wb') as model_file:
    pickle.dump(model,model_file)
print("Model has been saved as Customer_Purchase_Prediction_Model.pkl")