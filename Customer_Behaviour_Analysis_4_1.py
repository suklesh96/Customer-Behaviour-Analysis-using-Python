#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing Libraries.

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd

# EDA
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Pre-processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Clustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

import matplotlib.patheffects as path_effects
from matplotlib import colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from sklearn.cluster import AgglomerativeClustering

# SVM
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Sweetviz report
import sweetviz as sv

# Pandas profile
from pandas_profiling import ProfileReport

# Avoid warnings
import warnings
warnings.simplefilter("ignore")


# ### 2. Loading the data set

# In[2]:


# Load data
df = pd.read_csv("marketing_campaign.csv", sep = "\t")


# # Information about the data set

# People
# ●	ID: Customer's unique identifier
# ●	Year_Birth: Customer's birth year
# ●	Education: Customer's education level
# ●	Marital_Status: Customer's marital status
# ●	Income: Customer's yearly household income
# ●	Kidhome: Number of children in customer's household
# ●	Teenhome: Number of teenagers in customer's household
# ●	Dt_Customer: Date of customer's enrollment with the company
# ●	Recency: Number of days since customer's last purchase
# ●	Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# 
# Products
# ●	MntWines: Amount spent on wine in last 2 years
# ●	MntFruits: Amount spent on fruits in last 2 years
# ●	MntMeatProducts: Amount spent on meat in last 2 years
# ●	MntFishProducts: Amount spent on fish in last 2 years
# ●	MntSweetProducts: Amount spent on sweets in last 2 years
# ●	MntGoldProds: Amount spent on gold in last 2 years
# 
# Promotion
# ●	NumDealsPurchases: Number of purchases made with a discount
# ●	AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# ●	AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# ●	AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# ●	AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# ●	AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# ●	Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# Place
# ●	NumWebPurchases: Number of purchases made through the company’s website
# ●	NumCatalogPurchases: Number of purchases made using a catalogue
# ●	NumStorePurchases: Number of purchases made directly in stores
# ●	NumWebVisitsMonth: Number of visits to company’s website in the last month

# ### 3. EDA

# In[3]:


df.head()


# In[4]:


df.shape


# ### Pandas Profile Report:

# In[5]:


pr = ProfileReport(df)
pr


# In[6]:


# report by sweetviz

my_report = sv.analyze(df)
my_report.show_html()


# In[7]:


df.sample(5)


# In[8]:


# set up to view all the info of the columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[9]:


df.sample(5)


# In[10]:


options = ['Graduation' ,'PhD'] 
    
# selecting rows based on condition 
rslt_df = df[(df['ID'] == 4141) & 
          df['Education'].isin(options)] 
    
print('\nResult dataframe :\n',
      rslt_df)


# In[11]:


def basic_info(df):
    print("This dfset has ", df.shape[1], " columns and ", df.shape[0], " rows.")
    print("This dfset has ", df[df.duplicated()].shape[0], " duplicated rows.")
    print(" ")
    print("Descriptive statistics of the numeric features in the dfset: ")
    print(" ")
    print(df.describe())
    print(" ")
    print("Information about this dfset: ")
    print(" ")
    print(df.info())


# In[12]:


basic_info(df)


# In[13]:


df.median()


# In[14]:


df_copy = df.copy()


# In[15]:


# Divide the data into two dataframes: one has income values, and the other doesn't.
have_income = df_copy[df_copy.Income.isnull()== False]
missing_income = df_copy[df_copy.Income.isnull()== True]


# In[16]:


df_copy.info()


# In[17]:


# Convert the one that has income to int type
have_income.Income = have_income.Income.astype('int64')

# Give a string value of "0" to missing value, then we can convert it into int type
missing_income.Income = str(have_income.Income.median())

# Coverting String and Float dtypes to int dtype
missing_income.Income = missing_income.Income.str.replace(".5", "")
missing_income.Income = missing_income.Income.astype('int64')


# In[18]:


# Combine the data
df_copy = missing_income.append(have_income)


# In[19]:


options = ['Graduation' ,'PhD','2n Cycle','Master'] 
    
# selecting rows based on condition 
rslt_df = df_copy[(df_copy['ID'] == 5250) & 
          df_copy['Education'].isin(options)] 
    
print('\nResult dataframe :\n',
      rslt_df)


# In[20]:


df_copy.info()


# #### 3.1 Issue regarding date 

# In[21]:


# This function converts a scalar, array-like, Series or DataFrame/dict-like to a pandas datetime object.
df_copy.Dt_Customer = pd.to_datetime(df_copy.Dt_Customer)


# In[22]:


df_copy.info()


# #### 3.2 Reset the index

# In[23]:


# Reset the index
df_1 = df_copy.reset_index(drop=True)


# In[24]:


df_1.sample(5)


# In[25]:


df_1.info()


# ### 3.3 Visualizations

# #### 3.3.1 Finding and treating the outliers

# In[26]:


# select columns to plot
# Dropping the categorical columns/ un-useful columns(ID) --> to plot outliers
df_1_to_plot = df_1.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain','Z_CostContact','Z_Revenue']).select_dtypes(include=np.number)

# subplots: layout=(rows,columns) ; kind='Type of boxblot' ; patch_artist=True (To fill the boxplot with colour)
df_1_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(15,20), patch_artist=True)

plt.suptitle('Find Outliers', fontsize=15, y=0.9)
plt.savefig('boxplots.png', bbox_inches='tight')


# #### Treating outliers (using emperical formula (µ - 3σ) )

# In[27]:


df_1.Year_Birth.describe()


# In[28]:


# Remove outliers in year_birth -->(µ - 3σ)

df_1 = df_1[df_1.Year_Birth >= (df_1.Year_Birth.mean()-3*df_1.Year_Birth.std())]
df_1.Year_Birth.describe()


# In[29]:


# Outliers after treating on them
df_1_to_plot = df_1.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain','Z_CostContact','Z_Revenue']).select_dtypes(include=np.number)

# subplots: layout=(rows,columns) ; kind='Type of boxblot' ; patch_artist=True (To fill the boxplot with colour)
df_1_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(15,20), patch_artist=True)

plt.suptitle('Find Outliers', fontsize=15, y=0.9)
plt.savefig('boxplots.png', bbox_inches='tight')


# In[30]:


#Dropping some of the redundant features
to_drop = ["Z_CostContact", "Z_Revenue"]
df_1 = df_1.drop(to_drop, axis=1)


# In[31]:


# Outliers after treating on them
df_2_to_plot = df_1.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include=np.number)

# subplots: layout=(rows,columns) ; kind='Type of boxblot' ; patch_artist=True (To fill the boxplot with colour)
df_2_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(15,20), patch_artist=True)

plt.suptitle('Find Outliers', fontsize=15, y=0.9)
plt.savefig('boxplots.png', bbox_inches='tight')


# In[32]:


#Dropping the outliers by setting a cap on income. 
df_1 =df_1[(df_1["Income"]<600000)]
print("The total number points after removing the outliers are:", len(df_1))


# In[33]:


# Outliers after treatting on them
df_2_to_plot = df_1.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include=np.number)

# subplots: layout=(rows,columns) ; kind='Type of boxblot' ; patch_artist=True (To fill the boxplot with colour)
df_2_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(15,20), patch_artist=True)

plt.suptitle('Find Outliers', fontsize=15, y=0.9)
plt.savefig('boxplots.png', bbox_inches='tight')


# In[34]:


df_1.sample(5)


# In[35]:


df_1.info()


# In[36]:


#correlation matrix

cmap =ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

corrmat= df_1.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)


# In[37]:


# Run discriptive statistics of numerical datatypes.
df_1.describe(include = ['float64','int64'])


# In[38]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Education', data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[ ]:





# In[ ]:





# #### 3.3.2 Analysis -->
# a) Uni-Variate Analysis
# b) Bi-Variate Analysis
# c) Multi-Variate Analysis

# In[39]:


#Feature Engineering
#Age of customer today 
df_1["Age"] = 2021-df_1["Year_Birth"]

#Total spendings on various items
df_1["Total_Spent"] = df_1["MntWines"]+ df_1["MntFruits"]+ df_1["MntMeatProducts"]+ df_1["MntFishProducts"]+ df_1["MntSweetProducts"]+ df_1["MntGoldProds"]

#Deriving living situation by marital status"Alone"
df_1["Living_With"]=df_1["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
df_1["Children"]=df_1["Kidhome"]+df_1["Teenhome"]

#Feature for total members in the householde
df_1["Family_Size"] = df_1["Living_With"].replace({"Alone": 1, "Partner":2})+ df_1["Children"]

#Feature pertaining parenthood
df_1["Is_Parent"] = np.where(df_1.Children> 0, 1, 0)

#Segmenting education levels in three groups
df_1["Education"]=df_1["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
df_1=df_1.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})


# In[40]:


df_1["Join_year"] = df_1.Dt_Customer.dt.year
df_1["Join_month"] = df_1.Dt_Customer.dt.month
df_1["Join_weekday"] = df_1.Dt_Customer.dt.weekday

df_1['Total_num_purchase'] = df_1.NumDealsPurchases+ df_1.NumWebPurchases+ df_1.NumCatalogPurchases+ df_1.NumStorePurchases+ df_1.NumWebVisitsMonth 
df_1['Total_accept'] = df_1.AcceptedCmp1 + df_1.AcceptedCmp2 + df_1.AcceptedCmp2 + df_1.AcceptedCmp2  + df_1.AcceptedCmp3 + df_1.AcceptedCmp4 + df_1.AcceptedCmp5 + df_1.Response


# In[41]:


# Run discriptive statistics of numerical datatypes.
df_1.describe(include = ['float64','int64'])


# In[42]:


df_1["Income"].value_counts()


# In[43]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Education', data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[44]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Living_With', data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[45]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Children', data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[46]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Join_weekday', data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[47]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Join_month', data=df_1)
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[48]:


# Creating dataset
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

data = [191,187,202,184,191,170,141,211,164,209,185,202]


# Creating explode data
explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.25,0.3,0.0,0.0,0.0,0.0)

# Creating color parameters
colors = ( "orange", "cyan", "brown","grey", "indigo", "beige","blue","yellow","violet","pink","purple","green")

# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }

# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(data,
autopct = lambda pct: func(pct, data),
explode = explode,
labels = months,
shadow = True,
colors = colors,
startangle = 90,
wedgeprops = wp,
textprops = dict(color ="black"))

# Adding legend
ax.legend(wedges, months,
title ="Months",
loc ="center left",
bbox_to_anchor =(1.3, 0, 0.5, 1))

plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Customizing pie chart")

# show plot
plt.show()


# In[49]:


# Lets Display Count on top of countplot

fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Join_year', data=df_1)
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[50]:


# Combined univariate analysis of each variables
fig,axes = plt.subplots(2,4, figsize=(16,10))
sns.countplot('Education',data=df_1,ax=axes[0,0])
sns.countplot('Marital_Status',data=df_1,ax=axes[0,1])
sns.countplot('Children',data=df_1,ax=axes[0,2])
sns.countplot('Join_weekday',data=df_1,ax=axes[0,3])
sns.countplot('Join_month',data=df_1,ax=axes[1,0])
sns.countplot('Join_year',data=df_1,ax=axes[1,1])

#sns.distplot(df_train['Fare'], kde=True,ax=axes[1,2])
sns.histplot(data=df_1,x="Total_Spent",ax=axes[1,2])
sns.histplot(data=df_1,x="Income",ax=axes[1,3] )


# ### Bivariate Analysis

# We perform bi-variate analysis with 2 variables for any combination of categorical and continuous variables.
# The combination can be: Categorical & Categorical, Categorical & Continuous and Continuous & Continuous.

# In[51]:


# Lets more elaborate customer behaviour with Education and Income --> use catplot or countplot
'''
fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Education',hue="Minorhome",data=df_1)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
'''

sns.catplot(x="Education",col="Children",data=df_1, kind="count",height=4, aspect =.7)


# In[52]:


# Lets more elaborate customer behaviour with Education and Income --> use catplot or countplot

sns.catplot(x="Education",col="Join_weekday",data=df_1, kind="count",height=4, aspect =.7)


# In[53]:


# Lets more elaborate customer behaviour with Teenhome and Total_Spent --> use catplot or countplot
'''
fig, ax1 = plt.subplots(figsize=(5,5))
graph = sns.countplot(ax=ax1,x='Total_Spent',hue="Teenhome",data=new_df)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
'''

sns.catplot(x="Total_Spent",col="Teenhome",data=df_1, kind="count",height=4, aspect =.7)


# In[54]:


fig, ax1 = plt.subplots(figsize=(10,7))

def main():
    sns.set_style("whitegrid")
    tips = df_1
    # optionally disable fliers
    showfliers = True
    # plot data and create median labels
    box_plot = sns.boxplot(ax=ax1, x='Living_With', y='Income', hue='Education', data=df_1,
                           showfliers=showfliers)
    create_median_labels(box_plot.axes, showfliers)
    plt.show()


def create_median_labels(ax, has_fliers):
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4:len(lines):lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)
        # print text to center coordinates
        text = ax.text(mean_x, mean_y, f'{mean_y:.1f}',
                       ha='center', va='center',
                       fontweight='bold', size=10, color='white')
        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])


if __name__ == '__main__':
    main()


# In[55]:


fig, ax1 = plt.subplots(figsize=(10,7))

def main():
    sns.set_style("whitegrid")
    tips = df_1
    # optionally disable fliers
    showfliers = True
    # plot data and create median labels
    box_plot = sns.boxplot(ax=ax1, x='Children', y='Income', hue='Living_With', data=df_1,
                           showfliers=showfliers)
    create_median_labels(box_plot.axes, showfliers)
    plt.show()


def create_median_labels(ax, has_fliers):
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4:len(lines):lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)
        # print text to center coordinates
        text = ax.text(mean_x, mean_y, f'{mean_y:.1f}',
                       ha='center', va='center',
                       fontweight='bold', size=10, color='white')
        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])


if __name__ == '__main__':
    main()


# In[56]:


fig, ax1 = plt.subplots(figsize=(10,7))

def main():
    sns.set_style("whitegrid")
    tips = df_1
    # optionally disable fliers
    showfliers = True
    # plot data and create median labels
    box_plot = sns.boxplot(ax=ax1, x='Children', y='Total_Spent', hue='Family_Size', data=df_1,
                           showfliers=showfliers)
    create_median_labels(box_plot.axes, showfliers)
    plt.show()


def create_median_labels(ax, has_fliers):
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4:len(lines):lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)
        # print text to center coordinates
        text = ax.text(mean_x, mean_y, f'{mean_y:.1f}',
                       ha='center', va='center',
                       fontweight='bold', size=10, color='white')
        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])


if __name__ == '__main__':
    main()


# In[57]:


fig, ax1 = plt.subplots(figsize=(10,7))

def main():
    sns.set_style("whitegrid")
    tips = df_1
    # optionally disable fliers
    showfliers = True
    # plot data and create median labels
    box_plot = sns.boxplot(ax=ax1, x='Education', y='Total_Spent', hue='Join_year', data=df_1,
                           showfliers=showfliers)
    create_median_labels(box_plot.axes, showfliers)
    plt.show()


def create_median_labels(ax, has_fliers):
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4:len(lines):lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)
        # print text to center coordinates
        text = ax.text(mean_x, mean_y, f'{mean_y:.1f}',
                       ha='center', va='center',
                       fontweight='bold', size=10, color='white')
        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])


if __name__ == '__main__':
    main()


# In[58]:


education = df_1.Education.value_counts()

fig = px.pie(education, 
             values = education.values, 
             names = education.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# In[59]:


df_1.loc[(df_1['Age'] >= 13) & (df_1['Age'] <= 19), 'AgeGroup'] = 'Teen'
df_1.loc[(df_1['Age'] >= 20) & (df_1['Age']<= 39), 'AgeGroup'] = 'Adult'
df_1.loc[(df_1['Age'] >= 40) & (df_1['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
df_1.loc[(df_1['Age'] > 60), 'AgeGroup'] = 'Senior Adult'


# In[60]:


children = df_1.Children.value_counts()

fig = px.pie(children, 
             values = children.values, 
             names = children.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# In[61]:


childrenspending = df_1.groupby('Children')['Total_Spent'].mean().sort_values(ascending=False)
childrenspending_df_1 = pd.DataFrame(list(childrenspending.items()), columns=['No. of Children', 'Average Spending'])

plt.figure(figsize=(10,5))

sns.barplot(data=childrenspending_df_1,  x="No. of Children", y="Average Spending", palette='rocket_r');
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('No. of Children', fontsize=13, labelpad=13)
plt.ylabel('Average Spending', fontsize=13, labelpad=13);


# In[62]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = df_1.Age, color='salmon')
ax.set(title = "Age Distribution of Customers");
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Age ', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);


# In[63]:


def find_IQR(df, column):
  q_25, q_75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
  IQR = q_75 - q_25
  whiskers_range = IQR * 1.5
  lower, upper = q_25 - whiskers_range, whiskers_range + q_75
  return lower, upper


# In[64]:


lower_age, upper_age = find_IQR(df_1, "Age")
print(lower_age, upper_age)


# In[65]:


lower_income, upper_income = find_IQR(df_1, "Income")
print(lower_income, upper_income)


# In[66]:


def find_IQR(df_1, column):
  q_25, q_75 = np.quantile(df_1[column], 0.25), np.quantile(df_1[column], 0.75)
  IQR = q_75 - q_25
  whiskers_range = IQR * 1.5
  lower, upper = q_25 - whiskers_range, whiskers_range + q_75
  return lower, upper


# In[67]:


# Drop the outliers
df_1 = df_1[(df_1["Age"] < upper_age)]
df_1 = df_1[(df_1["Income"] < upper_income)]


# In[68]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = df_1.Age, color='salmon')
ax.set(title = "Age Distribution of Customers");
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Age ', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);


# The age of customers is almost normally distributed, with the majority of customers between the ages of 40 and 60.

# In[69]:


agegroup = df_1.AgeGroup.value_counts()

fig = px.pie(labels = agegroup.index, values = agegroup.values, names = agegroup.index, width = 550, height = 550)

fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.4, 
                  marker = dict(colors = ['#3D0C02', '#800000'  , '#C11B17','#C0C0C0'], 
                                line = dict(color = 'white', width = 2)))

fig.update_layout(annotations = [dict(text = 'Age Groups', 
                                      x = 0.5, y = 0.5, font_size = 20, showarrow = False,                                       
                                      font_color = 'black')],
                  showlegend = False)

fig.show()


# More than 50% of customers are Middle Age Adults between 40 and 60 The 2nd well-known age category is Adults, aged between 20 and 40

# In[70]:


products = df_1[['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']]
product_means = products.mean(axis=0).sort_values(ascending=False)
product_means_df_1 = pd.DataFrame(list(product_means.items()), columns=['Product', 'Average Spending'])

plt.figure(figsize=(15,10))
plt.title('Average Spending on Products')
sns.barplot(data=product_means_df_1, x='Product', y='Average Spending', palette='rocket_r');
plt.xlabel('Product', fontsize=20, labelpad=20)
plt.ylabel('Average Spending', fontsize=20, labelpad=20);


# ### 4. Data Pre-processing

# In[71]:


#Get list of categorical variables:

s = (df_1.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[72]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    df_1[i]= df_1[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")


# In[73]:


df_1.sample(5)


# Wine and Meat Products are the most famous products among customers Candy and Fruits are not often bought

# In[74]:


df_clust = df_1.drop(['ID', 'Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome','Dt_Customer', 'Recency', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
                         'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                        'AcceptedCmp1', 'AcceptedCmp2', 'Response','Is_Parent','Join_month','Join_weekday'], axis=1)


# In[75]:


df_clust.sample(7)


# ### 5. Dimensional Reduction

# In[76]:


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(df_clust)
PCA_ds = pd.DataFrame(pca.transform(df_clust), columns=(["col1","col2","col3"]))
PCA_ds.describe().T


# In[77]:


#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]

#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="green", marker="1") # c="colour of points" and marker = "shape of the points ex: circle,square.etc"
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()



# In[78]:



#Ploting result data with the use of scatterplot. plotly
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]

fig = go.Figure(data=[go.Scatter3d(
    x=x,y=y,z=z,mode='markers',
    marker=dict(size=6,color='green',opacity=0.8))])

# tight layout
fig.update_layout( title={'text': "3D scatterplot of size-reduced data",'y':0.9,
        'x':0.5,'xanchor': 'center','yanchor': 'top'},
                  margin=dict(l=200, r=220, b=0, t=0))
fig.show()


# In[79]:


#Ploting result data with the use of scatterplot. plotly
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]

 
fig = go.Figure(data=[go.Scatter3d(
    x=x,y=y,z=z,mode='markers',
    marker=dict(size=6,color=x,opacity=0.8))])

# tight layout
fig.update_layout( title={'text': "3D scatterplot of size-reduced data",'y':0.9,
        'x':0.5,'xanchor': 'center','yanchor': 'top'},
                  margin=dict(l=200, r=220, b=0, t=0))
fig.show()


# ### Elbow method

# In[80]:


# Elbow method:

fig = plt.figure(figsize=(13,7))
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k = 10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# In[81]:


model = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(df_clust)

preds = model.predict(df_clust)

customer_kmeans = df_clust.copy()
customer_kmeans['clusters'] = preds


# In[82]:


#Income
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Income',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Income', fontsize=50, labelpad=20);


# In[83]:


#Total Spending
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Total_Spent',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=50, labelpad=20);


# In[84]:


plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Age',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Age', fontsize=50, labelpad=20);


# In[85]:


plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Children',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('No. of Children', fontsize=50, labelpad=20);


# ### From the above analysis we can group customers into 4 groups based on their income and total expenses:
# 
# ### Platinum: Highest income and highest expense
# 
# ### Gold: High earners and high spenders
# 
# ### Silver: The one with lower salary and less expenses
# 
# ### Bronze: The one with the lowest salary and least expenses

# In[86]:


customer_kmeans.clusters = customer_kmeans.clusters.replace({ 0: 'Platinum',1: 'Bronze',
                                                             2: 'Gold',
                                                             3: 'Silver',
                                                             })

df_clust['clusters'] = customer_kmeans.clusters


# In[87]:


cluster_counts = df_clust.clusters.value_counts()

fig = px.pie(cluster_counts, 
             values = cluster_counts.values, 
             names = cluster_counts.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# Most customers are in the Silver and Gold categories, around 29% and 28% respectively
# Platinum is the 3rd well-known customer category with 24% while only 19% occupies the bronze category
# 

# Relationship: Income vs. Spendings

# In[88]:


plt.figure(figsize=(20,10))
sns.scatterplot(data=df_clust, x='Income', y='Total_Spent', hue='clusters', palette='rocket_r');
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Total Spendings', fontsize=20, labelpad=20);


# 4 clusters can be easily identified from the plot above
# Those who earn more also spend more

# Spending Habits by Clusters

# In[89]:


df_clust.sample(3)


# In[ ]:





# ### 6. Dimensional Reduction for Cluster formation

# In[90]:


customer_kmeans.clusters = df_clust.clusters.replace({'Platinum':0, 'Gold':1,'Silver':2,'Bronze':3})

df_clust['Clusters'] = customer_kmeans.clusters


# In[91]:


df_1.sample(5)


# In[92]:


df_clust.sample(5)


# In[93]:


final = df_clust.drop(['Wines','Fruits','Meat','Fish','Sweets','Gold','Age','clusters'],axis = 1)


# In[94]:


final.sample(3)


# In[95]:


final.info()


# In[96]:



final=final.assign(Incomes=pd.cut(final['Income'], 
                               bins=[ 0, 25000, 50000,100000,666666], 
                               labels=['Below 25000', 'Income 25000-50000 ', 'Income 50000-100000 ','Above 100000']))
final=final.drop("Income", axis=1)


# In[97]:



final=final.assign(Expense=pd.cut(final['Total_Spent'], 
                               bins=[ 0, 500, 1000, 2525], 
                               labels=['Below 500', 'Expense 500-1000 ','Above 1000']))
final=final.drop("Total_Spent", axis=1)


# In[98]:


final.sample(5)


# In[99]:


# Label Encoding:
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[100]:


final['Incomes']= label_encoder.fit_transform(final['Incomes'])
# final['DOB']= label_encoder.fit_transform(final['DOB'])
final['Expense']= label_encoder.fit_transform(final['Expense'])


# In[101]:


from sklearn.preprocessing import normalize
final_scaled = normalize(final)
final_scaled = pd.DataFrame(final_scaled, columns=final.columns)
final_scaled.sample(5)


# In[ ]:





# In[102]:


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(final_scaled)
PCA_ds = pd.DataFrame(pca.transform(final_scaled), columns=(["col1","col2","col3"]))
PCA_ds.describe().T


# # Model Building

# In[103]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[104]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward") 


# In[105]:


y_hc=hc.fit_predict(final_scaled)
clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[106]:


y_hc


# In[107]:


clusters.sample(10)


# In[108]:


final.sample(3)


# In[109]:


final.shape


# In[110]:


X = final.drop("Clusters", axis=1)
Y = final.Clusters
X.shape, Y.shape


# In[111]:


X.sample(5)


# In[112]:


Y.sample(8)


# In[113]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10)


# In[114]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # SVM

# ### Grid Search

# #### 1. Kernel rbf

# In[115]:


clf =SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[1,5,14,13,12,11,10,0.1,0.001]}]
gsv = GridSearchCV(clf, param_grid,cv=10)
gsv.fit(x_train, y_train)


# In[116]:


gsv.best_params_ , gsv.best_score_


# In[117]:


clf = SVC(C =14 , gamma= 0.5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc_1= accuracy_score(y_test, y_pred)*100
print('Accuracy changing the kernel as rbf is:', np.round(acc_1,2))


# In[118]:


confusion_matrix(y_test, y_pred)


# #### 2. Polynomial Kernel

# In[119]:


# clf2 = SVC()
# param_grid2 = [{'kernel':['poly'],'gamma':[50,5,0.5,0.2,0.1,0.05],'C':[15,14,22,30,37,48,12,15,27]}]
# gsv2 = GridSearchCV(clf2, param_grid2, cv=10)
# gsv2.fit(x_train,y_train)


# In[120]:


# gsv2.best_params_, gsv2.best_score_


# In[121]:


# clf2= SVC(C=15, gamma=0.05, kernel = 'poly')
# clf2.fit(x_train, y_train)
# y_pred = clf2.predict(x_test)
# acc_2 = accuracy_score(y_test, y_pred)*100
# print('Accuracy',np.round(acc_2,3))


# In[122]:


# confusion_matrix(y_test, y_pred)


# #### 3.Sigmoid Kernel

# In[123]:


clf3 = SVC()
param_grid3 = [{'kernel':['sigmoid'], 'gamma':[30,50,5,0.5,0.2,0.1,0.05],"C":[15,14,20,30,35,48,12,16,29]}]
gsv3 = GridSearchCV(clf3, param_grid3, cv =10)
gsv3.fit(x_train, y_train)


# In[124]:


gsv3.best_params_, gsv3.best_score_


# In[125]:


clf = SVC(C=14, gamma=0.05, kernel= 'sigmoid')
clf3.fit(x_train, y_train)
y_pred = clf3.predict(x_test)
acc_3 = accuracy_score(y_test, y_pred) * 100
print('Accuracy', np.round(acc_3,3))


# In[126]:


confusion_matrix(y_test, y_pred)


# In[127]:


print("Accuracy using rbf kernel: ",np.round(acc_1,2))
#print("Accuracy using Polynomial kernel: ",np.round(acc_2,2))
print("Accuracy using Sigmoid kernel: ",np.round(acc_3,2))


# # Random Forest Classifier

# In[128]:


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)


# In[129]:


x_train.shape


# In[130]:


x_test.shape


# In[131]:


from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_test)
accuracy_score(y_test,pred_cv)*100


# In[132]:


pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)*100


# In[133]:


# saving the model 
import pickle 
pickle_out = open("RFC_model_1.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# In[ ]:




