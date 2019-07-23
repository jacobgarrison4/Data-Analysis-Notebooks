
# coding: utf-8

# <h1>
# <center>
# Assignment 1: Wrangle that data
# </center>
# </h1>
# 
# I'll ask you to use the skills you learned working with the Titanic table on a new table. The new table is data collected by a bank on the loans the bank gave out over a short period. There are 614 seperate loan applications. Some were accepted ("survived" in Titanic terms) and some were rejected ("died"). Eventually we will explore several machine learning models that try to learn how to recognize winning applications from losing applications. For now, we are just trying to wrangle the data we recieve.
# <p>
# You will see the general style I use for homeworks (and exams) is to solve the problem first myself, then remove the code and leave you with the output. So you can see what you should produce. Your job is to fill the code back in that produces it.

# # Read in spreadsheet
# 
# For the loan table, I stored the csv file in google sheets. I got its url and pasted it below. That's all I had to do. In particular, no need to copy/upload the csv file to colab. Just read it directly off the cloud.

# In[8]:


import pandas as pd
url = 'https://docs.google.com/spreadsheets/d/1_artlzgoj6pDBCBfdt9-Jmc9RT9yLsZ0vTnk3zJmt_E/pub?gid=1291197392&single=true&output=csv'
loan_table = pd.read_csv(url)

len(loan_table)  #614


# In[9]:


#I am setting the option to see all the columns of our table as we build it.
pd.set_option('display.max_columns', None)


# # 1. Explore
# 
# * Use head to get general layout.
# 
# * Find which columns have NaNs (empties) and how many
# 
# * Use describe method to see if any odd looking columns, e.g., more than 2 unqiue values for a binary column

# In[10]:


#show first 5 rows
loan_table.head(n=5)


# In[11]:


#get a count of empties in each column
loan_table.isnull().sum()


# In[12]:


#show statistics on each column
loan_table.describe(include='all')


# # 2. Drop the Loan_ID column
# 
# It does not carry useful information.

# In[13]:


#drop the column
loan_table = loan_table.drop(['Loan_ID'], axis=1)


# In[14]:


loan_table.head(1)  # Should see drop of Loan_ID column


# # 3. Outliers
# 
# Compute 3-sigma for the `LoanAmount` column. You need low and high values.
# 
# Then show all rows lower than low value and higher than high value.

# In[15]:


sigma3_lam = loan_table['LoanAmount'].std() * 3
mean_lam = loan_table['LoanAmount'].mean()
low_lam = mean_lam - sigma3_lam
high_lam = mean_lam + sigma3_lam
print((low_lam,high_lam))   # (-110.34981354495417, 403.1741378692785)


# In[16]:


#Build sub-table of all rows with low_lam outliers - should be the empty table - no rows
loan_table.loc[loan_table['LoanAmount'] < low_lam]


# In[17]:


#Build sub-table of all rows with high_lam outliers - should have 14 rows
loan_table.loc[loan_table['LoanAmount'] > high_lam]


# What looks suspicious to me are rows 177 and 523 with low `ApplicantIncome`. Unlikely someone will ask for a big loan with small income. However, both rows compensate by having `CoapplicantIncome` to fall back on.

# # 4. Handle empties (part 1)
# 
# Focus on the `LoanAmount` column. First build a new column that shows empties. Call it `no_lam`.
# 
# Then build another column and fill empties with mean. Call this new column `filled_lam`.

# In[18]:


#build no_lam column
loan_table['no_lam'] = loan_table.apply(lambda row: 1 if pd.isnull(row.LoanAmount) else 0, axis=1)
loan_table.head(1)


# In[19]:


# Build  filled_lam column
loan_table['filled_lam'] = loan_table.apply(lambda row: mean_lam if pd.isnull(row.LoanAmount) else row.LoanAmount, axis = 1)
loan_table.head(1)


# # 4. Handle empties (part 2)
# 
# Now focus on `Property_Area`. Use one-hot encoding to generate 4 new columns. Why 4? Because there are empties in this column. Prefix the columns with `pa`.

# In[20]:


#do one-hot encoding to get 4 new columns
ohe_p = pd.get_dummies(loan_table['Property_Area'], prefix='pa', dummy_na=True)
loan_table = loan_table.join(ohe_p)
loan_table.head(1)


# # 5. Bin continuous columns
# 
# Focus on `filled_lam`. Create a new column `lam_bin` that bins the values into 3 bins Low, Average, High. Allow the `cut` operator to choose the boundaries for you.

# In[21]:


#add the new column that contains the 3 bins.
bins = 3
bin_names = ['Low', 'Avergae', 'High']
loan_table['lam_bin'] = pd.cut(loan_table['filled_lam'], bins, labels=bin_names)
loan_table['lam_bin'].value_counts()


# # 6. One more one-hot encoding
# 
# Use one-hot encoding on `lam_bin`. You do *not* need to include an empty column given you have already captured that info in no_lam. So you should have 3 new columns generated.

# In[22]:


#do ohe on the lam_bin column
ohe_lam = pd.get_dummies(loan_table['lam_bin'], prefix='lam', dummy_na=False)
loan_table = loan_table.join(ohe_lam)
loan_table.head(1)


# # 7. Write it out
# 
# The next assignment will start with what you have done here. So write it out so you can read it back in next assignment.

# In[23]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[26]:


#now write it
import os
week = 1
home_path = os.path.expanduser('~')
file_path = '/Desktop/Cis 399/'
file_name = 'loan_wrangled_w'+str(week)+'.csv'
loan_table.to_csv(home_path + file_path + file_name, index=False)


# In[27]:


test_table = pd.read_csv(home_path + file_path + file_name)
test_table.head(n=1)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script loan_assignment_p')

