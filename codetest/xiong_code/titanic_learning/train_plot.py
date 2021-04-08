import warnings
warnings.filterwarnings('ignore')

#core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
# for dirname, _, filenames in os.walk('.'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
# check the data

# print(train.head())
# print(train.shape)
# print(train.info())

# show the age distribution
# print('Avgerage. Age:',train['Age'].mean())
# plt.figure(figsize=(8,4))
# fig=sns.distplot(train['Age'],color='darkorange')
# fig.set_xlabel('Age',size=15)
# fig.set_ylabel('Density of Passengers',size=15)
# plt.title('Passenger Age Distribution',size=20)
# plt.show()

# show the fare distribution
# print('Avgerage. Fare : ',train['Fare'].mean())
# plt.figure(figsize=(8,4))
# fig = sns.distplot(train['Fare'], color="darkorange")
# fig.set_xlabel("Fare",size=15)
# fig.set_ylabel("Density of Passengers",size=15)
# plt.title('Titanic Fare Distribution',size = 20)
# plt.show()

#WE PLOT THE PASSENGER AGE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Pclass'],train['Age'],palette='Blues')
# fig.set_xlabel("Passenger Class",size=15)
# fig.set_ylabel("Age of Passenger",size=15)
# plt.title('Age Distribution/Pclass',size = 20)
# plt.show()


#WE PLOT THE PASSENGER AGE DISTRIBUTION VS PASSENGER Gender ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Sex'],train['Age'],palette='Blues')
# fig.set_xlabel("Gender",size=15)
# fig.set_ylabel("Age of Passenger",size=15)
# plt.title('Age Distribution/Gender',size = 20)
# plt.show()


#WE PLOT THE PASSENGER AGE DISTRIBUTION VS SURVIVED ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Survived'],train['Age'],palette='Blues')
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("Age of Passenger",size=15)
# plt.title('Age Distribution/Survived',size = 20)
# plt.show()

#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Pclass'],train[train['Fare']<=300]['Fare'],palette='Reds')
# fig.set_xlabel("Passenger Class",size=15)
# fig.set_ylabel("Fare",size=15)
# plt.title('Fare Distribution/Pclass',size = 20)
# plt.show()

#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Survived'],train[train['Fare']<=300]['Fare'],palette='Reds')
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("Fare",size=15)
# fig.set(xticklabels=["0-No","1-Yes"])
# plt.title('Fare(Scaled) Distribution/Survived',size = 20)
# plt.show()

#WE PLOT THE FARE DISTRIBUTION VS PASSENGER CLASS ON TITANIC
# plt.figure(figsize=(8,4))
# fig=sns.boxplot(train['Survived'],train['Pclass'],palette='Reds')
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("Fare",size=15)
# fig.set(xticklabels=["0-No","1-Yes"])
# plt.title('Fare(Scaled) Distribution/Survived',size = 20)
# plt.show()

# plt.figure(figsize=(8,4))
# fig=sns.violinplot(train["Age"],train["Sex"], hue=train["Survived"],split=True,palette='Reds')
# fig.set_ylabel("Sex",size=15)
# fig.set_xlabel("Age",size=15)
# plt.title('Age and Sex vs Survived',size = 20)
# plt.show()

# plt.figure(figsize=(8,4))
# fig=sns.violinplot(train["Pclass"],train['Age'], hue=train["Survived"],split=True,palette='Blues')
# fig.set_xlabel("Pclass",size=15)
# fig.set_ylabel("Age",size=15)
# plt.title('Age and Pclass vs Survived',size = 20)
# plt.show()

# bg_color = (0.25, 0.25, 0.25)
# sns.set(rc={"font.style":"normal",
#             "axes.facecolor":bg_color,
#             "figure.facecolor":bg_color,"text.color":"white",
#             "xtick.color":"white",
#             "ytick.color":"white",
#             "axes.labelcolor":"white"})
# plt.figure(figsize=(8,4))
# fig=sns.countplot(train['Survived'],hue=train['Pclass'],palette='Blues',saturation=0.8)
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("#",size=15)
# fig.set(xticklabels=["0-No","1-Yes"])
# plt.title('# of Survived/PClass',size = 20)
# plt.show()

# plt.figure(figsize=(8,4))
# fig=sns.countplot(train['Survived'],hue=train['Sex'],palette='Oranges',saturation=0.8)
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("#",size=15)
# fig.set(xticklabels=["0-No","1-Yes"])
# plt.title('# of Survived/Sex',size = 20)
# plt.show()

# plt.figure(figsize=(8,4))
# fig=sns.countplot(train['Survived'],hue=train['SibSp']>0,palette='Blues',saturation=0.8)
# fig.set_xlabel("Survived",size=15)
# fig.set_ylabel("#",size=15)
# fig.set(xticklabels=["0-No","1-Yes"])
# plt.title('# of Survived per Siblings/Spouses Onboard',size = 20)
# plt.show()

sns.set(rc={"font.style":"normal",
            "axes.facecolor":"white",
            "figure.facecolor":"white","text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black"})
plt.figure(figsize=(8,4))
fig=sns.countplot(y=train['Pclass'],hue=train['SibSp']>0,palette='Blues',saturation=1.0)
fig.set_xlabel("#",size=15)
fig.set_ylabel("Passenger Class",size=15)
plt.title('# of Pclass per Siblings/Spouses Onboard',size = 20)
plt.show()