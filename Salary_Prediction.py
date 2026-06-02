
# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle

# In[2]:
df = pd.read_csv(r"C:\Users\jayes\jay - projects\Salary_prediction project\salary_prediction dataset\Salary_Data (1).csv")
df

# In[3]:
df.info()

# In[4]:
df.isnull().sum ()

# In[5]:
df = df.dropna()
df

# In[6]:
df.describe()

# In[7]:
df.columns

# In[8]:
df["Education Level"].unique()

# In[9]:
df["Education Level"] = df["Education Level"].apply(lambda x: "Bachelor's" if x == "Bachelor's Degree" else x)
df["Education Level"] = df["Education Level"].apply(lambda x: "Master's" if x == "Master's Degree" else x)
df["Education Level"] = df["Education Level"].apply(lambda x: "PhD" if x == "phD" else x)

# In[10]:
df["Education Level"].unique()

# In[11]:
df["Job Title"].unique()


# In[12]:
encode = {}

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encode[col] = le

# In[13]:
X = df.drop(columns = ["Salary","Age"])
y = df["Salary"]

# In[14]:
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# In[15]:
model = RandomForestRegressor(n_estimators = 100)
model.fit(X_train,y_train)

# In[16]:
pred = model.predict(X_test)
pred

# In[17]:
mae = mean_absolute_error(y_test,pred)
mae

# In[18]:
df["Salary"].mean()

# In[19]:
fi = model.feature_importances_

# In[20]:
plt.barh(X.columns,fi)

# In[21]:
model_data = {
    "model": model,
    "encode":encode}

# In[22]:
with open("Salary_prediction.pkl","wb") as file:
    pickle.dump(model_data,file)
print("saved")

# In[23]:
