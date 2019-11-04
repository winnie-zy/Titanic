#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Load the training data
df_train=pd.read_csv('train.csv')
pd.set_option('display.max_columns',500)#设置pandas中display的列可有500个
#Check data shape
print(df_train.shape)

#check first five rows
df_train.head()

#Check missing data
df_train.isnull().sum()

#Fill in the missing age with median age number
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

#Only 2 Embarked is missing. 可以采用以下两种方式：1.Let's remove these two rows
df_train['Embarked'].unique()
df_train=df_train[df_train['Embarked'].notna()]

#2.Another way is to replace these two with the most frequent one 'S'
# sns.distplot(df_train['SalePrice']) 不能使用distplot，Embarked为string
df_train['Embarked'].hist()#查看Embarked数值的分布,找到分布最多的点，可用于fillna()
plt.show()
df_train['Embarked']=df_train['Embarked'].fillna('S')
#或是直接使用df_train['Embarked'].fillna('S'，inplace=True)
#inplace : bool, default False.If True, fill in-place. Note: this will modify any other views on this object (e.g., a no-copy slice for a column in a DataFrame).

#Convert categorical variable into dummy/indicator variables.
#将categorical variable 转变成 numeric data(dummy/indicator variables)
embarked_dummies = pd.get_dummies(df_train['Embarked'], prefix='Embarked')
df_train = pd.concat([df_train, embarked_dummies], axis=1)#axis=1 按列操作,将embarked_dummies连接到df_train上
df_train.drop('Embarked', axis=1, inplace=True)#DROP掉原来的Embarked列

#There are too many missing values in Cabin, instead of removing this variable, we use 'Missing' to replace it
df_train['Cabin'].fillna('Missing', inplace=True)
#或者使用df_train['Cabin']=df_train['Cabin'].fillna('missing')

#Also, the cabin contains numbers, but we care more about the Cabin class, so we will remove numbers.
# mapping each Cabin value with the cabin letter
#map是一种让函数作用于Series每一个元素的操作
df_train['Cabin'] = df_train['Cabin'].map(lambda x: x[0])
df_train['Cabin'].unique()

#Let's first check if correlation still works for this project
corrmat = df_train.corr()
sns.heatmap(corrmat, square=True,annot=True)  
plt.show()

#Let plot the survived data and other features
df_train['Died'] = 1 - df_train['Survived'] 
#Pclass  df_train.groupby('Pclass').sum()：对Pclass分组，然后计算每个列的总数，[['Survived', 'Died']]对其中两列进行计数
df_train.groupby('Pclass').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, colors=['g', 'r'])
#Sex
df_train.groupby('Sex').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, colors=['g', 'r'])
#Age variable
#A nice way to compare distributions is to use a violin plot
sns.violinplot(x='Sex', y='Age', hue='Survived', data=df_train, split=True,palette={0: "r", 1: "g"})

#SibSp
df_train.groupby('SibSp').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, colors=['g', 'r'])
#Cannot tell if SibSp matters becasue when SibSp >=2 the volumn is too small. We th en choose to plot percenage.
SibSp_df=df_train[['SibSp','Survived', 'Died']].groupby('SibSp').sum()
SibSp_df['Survive_R']=SibSp_df['Survived']/(SibSp_df['Survived']+SibSp_df['Died'])
SibSp_df['Died_R']=1-SibSp_df['Survive_R']
SibSp_df.groupby('SibSp').sum()[['Survive_R', 'Died_R']].plot(kind='bar', stacked=True, colors=['g', 'r'])

#Similarly, we will do the same for Parch
df_train.groupby('Parch').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, colors=['g', 'r'])
Parch_df=df_train[['Parch','Survived', 'Died']].groupby('Parch').sum()
Parch_df['Survive_R']=Parch_df['Survived']/(Parch_df['Survived']+Parch_df['Died'])
Parch_df['Died_R']=1-Parch_df['Survive_R']
Parch_df.groupby('Parch').sum()[['Survive_R', 'Died_R']].plot(kind='bar', stacked=True, colors=['g', 'r'])

#Ticket fare
plt.hist([df_train[df_train['Survived'] == 1]['Fare'], df_train[df_train['Survived'] == 0]['Fare']], stacked=True, color = ['g','r'], bins = 50, label = ['Survived','Died'])
plt.legend()

#Before we move futher, let's chech the correlation between input features
df_train.groupby('Pclass').mean()['Fare'].plot(kind='bar')
df_train.groupby('Pclass').mean()['Age'].plot(kind='bar')#age与Pclass的关系
#the upper part is to explore these data and find some suspecious relationship between output and inputs.

#Ok, now we have got a general idea about our datasets. It is time to do feature engineering and selection
# reading test data
df_test = pd.read_csv('test.csv')

# extracting and then removing the targets from the training data 
targets = df_train['Survived']
df_train.drop(['Survived'], 1, inplace=True)
    
# merging train data and test data for future feature engineering
# we'll also remove the PassengerID since this is not an informative feature
combined = df_train.append(df_test)#按照列拼接，叠加行
# combined.reset_index(inplace=True)#将index整合成一个整体,新增一列index
#drop=True:Do not try to insert index into dataframe columns. This resets the index to the default integer index.
combined.reset_index(drop=True)#将index整合成一个整体,不会新增index列
combined.drop(['index', 'PassengerId'], inplace=True, axis=1)#去掉对模型无用的coloumn

#Check the combined dataset shape
print(combined.shape)

#check the first five rows in combined dataset
combined.head()

#get unique titles from our combined datasets. strip function is to remove the extra spaces.
#check the unique and useful value from name column,and finally form the  Title_Dictionary
titles = set()
for name in combined['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)

#Now let's map the title can bin them
#Captain, Colonel, Major, Doctor, Reverend can be binned to officer
#Jonkheer, Don, Dona, Sir, the Countess, Lady can be binned to Royalty
#Madame, Ms, Mrs can be binned to Mrs
#Mademoiselle, , Miss can be binned to Miss
#Mr
#Master: male children: Young boys were formerly addressed as "Master [first name]."

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

#Generate a new Title column
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())#找出名字中的头衔
combined['Title'] = combined['Title'].map(Title_Dictionary)

#check if there is any missing Title
combined[combined['Title'].isnull()]

#check missing age data in the training and test dataset
print(combined.iloc[:891]['Age'].isnull().sum())#train,取前891行
print(combined.iloc[891:]['Age'].isnull().sum())#test

#let's get the median age based on people's gender, Pclass and Title
#基于training来做分析，然后apply到both train and test dataset.
grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
#也可以通过'Sex','Pclass','Title'分组，只算Age。combined.iloc[:891].groupby(['Sex','Pclass','Title']).median()['Age']
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
print(grouped_median_train)

#Now we just need to map these medium ages to the missing parts. [0] is to convert list to number.
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]
##此处不能用combined.map(),因为'DataFrame' object has no attribute 'map'
##如果改成combined['Age'].map(),调用fill_age()会报错，是对行的各个字段进行判断。此处只能用df.apply()
combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
print(combined['Age'].isnull().sum())

# Name can be dropped now
combined.drop('Name', axis=1, inplace=True)

# encoding in dummy variable,把convert categorical variable to dummy variables 
titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, titles_dummies], axis=1)

# removing the title variable
combined.drop('Title', axis=1, inplace=True)

#Fill out the missing fare data
#因为只有一行，可以直接用mean值填充，或是用之前group by('Pclass')精确计算
combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)

# two missing embarked values - filling them with the most frequent one in the train set
combined['Embarked'].unique()
df_train['Embarked'].hist()#查看Embarked数值的分布,找到分布最多的点，可用于fillna()
plt.show()
combined['Embarked'].fillna('S', inplace=True)

# encoding in dummy variable,把convert categorical variable to dummy variables
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)

#Now let's test if there are different Cabin categories in the test dataset
#Why? If there are some Cabin categories in the test dataset but not in the training dataset, we have to replace them.
#如果出现了在test里的，而train里没有的cabin值，需要在train_cabin里新增一个变量和test_cabin一致，保证Cabin categories在train,test一致
#train中多了cabin无所谓，这样的模型更robust，放到test上不会出错
#一下是确认过程
train_cabin, test_cabin = set(), set()
#training dataset
for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('M')
#test dataset    
for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('M')
print(train_cabin)
print(test_cabin)

#Now let's fill out the missing values for Cabin
combined['Cabin'].fillna('M', inplace=True)
combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

# dummy encoding ...get_dummies()将categorical data 转换成numerical data,数字用于模型公式的计算
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
combined = pd.concat([combined, cabin_dummies], axis=1)
combined.drop('Cabin', axis=1, inplace=True)

#For now, check missing data
combined.isnull().sum()

# encoding into 3 categories:
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
# adding dummy variable
combined = pd.concat([combined, pclass_dummies],axis=1)
    
# removing "Pclass"
combined.drop('Pclass',axis=1,inplace=True)

# mapping gender to numerical one 
combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

#Previously we have explored the SibSp and Parch, now we will merge these two together
# introducing a new feature : the size of families (including the passenger)
#无论是父母，小孩还是其他家庭成员对逃生的影响应该是相似的，成员越多越没有逃生动力
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
# introducing other features based on the family size
combined['Single'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

#a function that extracts each prefix of the ticket, returns 'NONE' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')#./直接被去掉
    ticket = ticket.replace('/','')
    ticket = ticket.split()#拆分成list,字母和数字在ticket中总是有空格，被空格分成多个元素如['A5', '21171']
    #map函数的原型是map(function, iterable, …),是将function应用于iterable的每一个元素，结果以列表的形式返回.
    #可以传很多个iterable，如果有额外的iterable参数，并行的从这些参数中取元素，并调用function。
    ticket = map(lambda t : t.strip(), ticket)#ticket目前是list<String>,每个元素前后去掉空格
    #list comprehensions returns lists， they consist of brackets containing the expression, 
    # which is executed for each element along with the for loop to iterate over each element.
    #List Comprehensions语法：[expr for iter_var in iterable] 或 [expr for iter_var in iterable if cond_expr]
    ticket = [x for x in ticket if not x.isdigit()]
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'NONE'##全数字的票

#Get Ticket info
combined['Ticket'] = combined['Ticket'].map(cleanTicket)

# Extracting dummy variables from tickets:
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)

#Check current dataset
print(combined.shape)

#Prepare the training dataset
df_im_input=combined.iloc[:891]
df_im_output=targets
targets.shape#得到(891,)表示891行，1列

#Now let's get the importance of each feature
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(df_im_input, df_im_output)
#通过去掉某些feature,模型的performance发生变化，
#通过比对确定每个feature的importance,不同input的导致模型accuracy不同
features = pd.DataFrame()
features['feature'] = df_im_input.columns#columns名称赋值给
#clf.feature_importances_得到每个feature的inportance
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)

#plot it
features.plot(kind='barh', figsize=(25, 25))

#select top 10 important features
top_10_feature=features.nlargest(10, 'importance')
#select top 15 important features(以下选取15个feature)
top_15_feature=features.nlargest(15, 'importance')
#choose model final input features
df_input_final=df_im_input[top_10_feature['feature']]
df_input_final_15=df_im_input[top_15_feature['feature']]

#build Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# logreg.fit(df_input_final,targets)
logreg.fit(df_input_final_15,targets)

#Let’s first test our training dataset prediction accuracy
#get predictions based on training input
preds=logreg.predict(df_input_final)
preds_15=logreg.predict(df_input_final_15)
preds_probabilities = logreg.predict_proba(df_input_final)
preds_probabilities_15 = logreg.predict_proba(df_input_final_15)

#preds_probabilities has two numbers for each row of features: [probability of false, probability of true]
preds_probabilities.shape

#just need one as they can be calcualted using 1- other
pred_probs = preds_probabilities[:, 1]
pred_probs_15 = preds_probabilities_15[:, 1]


from sklearn.metrics import roc_curve, auc
#roc_curve() returns a list of false positive rates (FPR) and true positives rates (TPR) for different configurations of the classifier used to plot the ROC.
[fpr, tpr, thr] = roc_curve(targets, pred_probs)
[fpr_15, tpr_15, thr_15] = roc_curve(targets, pred_probs_15)

#fpr:x,false  positive ;tpr:y,true positive
#plot ROC curve
plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot(fpr_15, tpr_15, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr_15, tpr_15))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)', fontsize=14)
# plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.xlabel('False Positive Rate (FPR_15)', fontsize=14)
plt.ylabel('True Positive Rate (TPR_15)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#check model accuracy on training dataset
from sklearn.metrics import confusion_matrix, accuracy_score
#targets:actual results,preds:predictive results
print("accuracy: %2.3f" % accuracy_score(targets, preds))#结果是0.820
print("AUC: %2.3f" % auc(fpr, tpr))#结果是0.872
print("accuracy: %2.3f" % accuracy_score(targets, preds_15))#结果是0.822
print("AUC: %2.3f" % auc(fpr_15, tpr_15))#结果是0.873
##为什么AUC比accuracy正确率高，因为AUC用的是可能率而不是0/1,这样的计算结果更加准确

#confusion matrix can give us the number of true positives, false positives, true negatives, and false negatives.
conf_m=confusion_matrix(targets, preds) 
conf_m_15=confusion_matrix(targets, preds_15) 

#get the test input and predictions
df_test_input_final=combined.iloc[891:][top_10_feature['feature']]
df_test_preds=logreg.predict(df_test_input_final)
df_test_input_final_15=combined.iloc[891:][top_15_feature['feature']]
df_test_preds_15=logreg.predict(df_test_input_final_15)
#output the results to a csv file
submit = pd.DataFrame()
test = pd.read_csv('test.csv')
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = df_test_preds
submit.to_csv('Titanic_LR.csv', index=False)

submit_15 = pd.DataFrame()
test = pd.read_csv('test.csv')
submit_15['PassengerId'] = test['PassengerId']
submit_15['Survived'] = df_test_preds
submit_15.to_csv('Titanic_LR_1.csv', index=False)
#15 features
#select top 15 important features
top_15_feature=features.nlargest(15, 'importance')
df_input_final=df_im_input[top_15_feature['feature']]

#build Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(df_input_final,targets)

#get predictions based on training input
preds=logreg.predict(df_input_final)
preds_probabilities = logreg.predict_proba(df_input_final)
pred_probs = preds_probabilities[:, 1]

[fpr, tpr, thr] = roc_curve(targets, pred_probs)

#plot ROC curve
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#check model accuracy on training dataset
from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy: %2.3f" % accuracy_score(targets, preds))
print("AUC: %2.3f" % auc(fpr, tpr))

conf_m=confusion_matrix(targets, preds) 





