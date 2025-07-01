import streamlit as st
import pandas as pd

st.set_page_config(page_title='Streamlit App')
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


df = pd.read_csv("sample.csv", encoding="ISO-8859-1")


# In[29]:


try:
    df = pd.read_csv("sample.csv", encoding="ISO-8859-1")
    st.write("تم تحميل البيانات بنجاح.")
except Exception as e:
    st.write("حدث خطأ أثناء تحميل البيانات:", e)
df


# In[8]:


df.isna().sum()


# In[9]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates()


# catorigal data

# In[10]:


df.info()


# In[11]:


df_cat=df[['Ship Mode','Customer Name','Customer ID','Segment','Country','City','State','Region','Product ID' ,'Category','Sub-Category','Product Name']]


# In[12]:


df_cat.head()


# In[13]:


sns.countplot(data=df, x='Category')
plt.title("catogry")
plt.xticks(rotation=45)
plt.show()


# In[14]:


## Top Selling and Profitable Products
product_group = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False)
top_5_selling = product_group.head(5)

top_5_selling.plot(kind="bar", title="Top 5 Selling Products", ylabel="Total Sales")
plt.xticks(rotation=45)
plt.show()


# In[15]:


for feature in df_cat.columns:
    st.write(feature,':',df[feature].nunique())


# In[16]:


df['Order Date'].nunique()


# In[17]:


df['Ship Date'].nunique()


# In[18]:


##what are top selling producting


# In[19]:


product_group =df.groupby(["Product Name"]).sum()['Sales']


# In[20]:


product_group


# In[21]:


top_selling_product=product_group.sort_values(ascending=False)


# In[22]:


Top_5_5selling_producting=pd.DataFrame(top_selling_product[:5])


# In[23]:


Top_5_5selling_producting


# In[24]:


Top_5_5selling_producting.plot(kind="bar")
plt.title("Top 5 selling")
plt.xlabel("product name")
plt.ylabel("total profit")
plt.show()


# In[25]:


## top profit product


# In[26]:


product_group =df.groupby(["Product Name"]).sum()['Profit']


# In[27]:


top_profit_product=product_group.sort_values(ascending=False)


# In[28]:


Top_5_profit_producting=pd.DataFrame(top_profit_product[:5])


# In[29]:


Top_5_profit_producting


# In[30]:


Top_5_profit_producting.plot(kind="bar")
plt.title("Top 5 profit")
plt.xlabel("product name")
plt.ylabel("total profit")
plt.show()


# In[31]:


##compare 


# In[32]:


Top_5_5selling_producting.index == Top_5_profit_producting.index


# In[33]:


fig,(axis1,axis2)= plt.subplots(1,2,figsize=(15,5))
Top_5_5selling_producting.plot(kind="bar", y="Sales", ax=axis1)
axis1.set_title ("Top 5 selling ")
Top_5_profit_producting.plot(kind="bar", y='Profit', ax=axis2)
axis2.set_title ("Top 5 profit ")
plt.show()                               


# In[34]:


st.write(Top_5_profit_producting.columns)


# In[35]:


df .Region.value_counts()


# In[37]:


import matplotlib.pyplot as plt
region_group = df.groupby("Region")[["Sales", "Profit"]].mean()
region_group.plot(kind="bar", figsize=(8, 5), colormap="Set2")
plt.title("Average Sales and Profit by Region")
plt.ylabel("Amount")
plt.xlabel("Region")
plt.grid(True)
plt.tight_layout()
plt.show()



# In[38]:


product= df[df["Product Name"]=="Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind"]
region_group = product.groupby("Region")[["Sales", "Profit"]].mean()
region_group.plot(kind="bar")
plt.show()


# In[39]:


product = df[(df["Product Name"] == "Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind") & (df["Region"] == 'Central')]
product["Discount"].plot(kind="bar")
plt.show()


# In[40]:


product = df[(df["Product Name"] == "Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind") & (df["Region"] == 'Central')]


# In[41]:


##what is the sales trend over time


# In[42]:


monthly_sales =df.groupby(['Order Date'],as_index=False).sum()


# In[44]:


monthly_sales


# In[45]:


df['Order Date'] = pd.to_datetime(df['Order Date'])


monthly_sales = df.set_index('Order Date')


monthly_sales = monthly_sales.resample("ME").sum() 
plt.figure(figsize=(25,5))
monthly_sales['Sales'].plot(kind='line')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# In[46]:


yearly_sales = monthly_sales.resample('Y').sum() 


plt.figure(figsize=(25,8))
plt.plot(yearly_sales['Sales'])
plt.xlabel("Order Date")
plt.ylabel("Sales")
plt.title("yearly Sales Trend")
plt.show()


# In[47]:


df['Profit Margin'] = df['Profit'] / df['Sales']

# Group the data by product category and calculate the average profit margin for each category
avg_profit_margin_by_category = df.groupby('Category')['Profit Margin'].mean()

# Plot the average profit margin for each category as a bar chart
avg_profit_margin_by_category.plot(kind='bar')

# Add a title and labels to the chart
plt.title("Average Profit Margin by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Profit Margin")

plt.show()


# In[48]:


df['Order Date'] = pd.to_datetime(df['Order Date'])

# تجميع المبيعات شهريًا
sales_by_month = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
sales_by_month.index = sales_by_month.index.to_timestamp()

plt.figure(figsize=(12, 6))
sales_by_month.plot()
plt.title('improve profit ')
plt.xlabel('date')
plt.ylabel('profit')
plt.grid(True)
plt.show()


# In[49]:


df.columns


# In[50]:


## Which region and place generate the most sales  


# In[51]:


df_places =df[['Country', 'City', 'State','Region','Sales','Profit']]


# In[52]:


df_places.head()


# In[53]:


for place in df_places.columns:
    st.write(place, ':', df_places[place].nunique())


# In[54]:


group_data =df_places.groupby(['Region'],as_index=False).sum()
group_data.sort_values(by='Sales', ascending=False, inplace=True)
# رابعاً: الرسم البياني
plt.figure(figsize=(10,5))
plt.bar(group_data['Region'], group_data['Sales'], align='center')

plt.xlabel('Region')
plt.ylabel('Sales')
plt.title("Sales genetered by state ")
plt.xticks(rotation =90)
plt.show()


# In[55]:


st.write(df_places.columns)



# In[56]:


group_data =df_places.groupby(['State'],as_index=False).sum()
group_data.sort_values(by='Sales', ascending=False, inplace=True)
# رابعاً: الرسم البياني
plt.figure(figsize=(10,5))
plt.bar(group_data['State'], group_data['Sales'], align='center')

plt.xlabel('State')
plt.ylabel('Sales')
plt.title("Sales genetered by state ")
plt.xticks(rotation =90)
plt.show()


# In[ ]:





# In[58]:


pivot_table = df.pivot_table(values=['Sales', 'Profit'], index='Region', aggfunc='mean')
plt.figure(figsize=(15, 6))
pivot_table.plot(kind='bar', stacked=False)
plt.title("Average Sales and Profit by Region")
plt.ylabel("Amount")
plt.xlabel("Region")
plt.grid(True)
plt.tight_layout()
plt.show()



# In[59]:


group_data =df_places.groupby(['City'],as_index=False).sum()
group_data.sort_values(by='Sales', ascending=False, inplace=True)
top_5_cities=group_data.head()
# رابعاً: الرسم البياني
plt.figure(figsize=(10,5))
plt.bar(top_5_cities['City'], top_5_cities['Sales'], align='center')

plt.xlabel('City')
plt.ylabel('Sales')
plt.title("top_5_cities by City ")
plt.xticks(rotation =90)
plt.show()


# In[60]:


top_5_cities=group_data.head()


# In[61]:


##what is the impact of discount on sales


# In[62]:


discount_group = df.groupby(["Discount"]).mean(numeric_only=True)[["Sales"]]
ax = discount_group.plot(kind="bar")
ax.set_ylabel("Sales")
plt.title("Average Sales by Discount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[63]:


discount_group


# In[65]:


plt.scatter (df['Discount'],df['Sales'])
plt.xlabel('Discount')
plt.ylabel('Sales')
plt.show()


# In[66]:


plt.scatter (df['Discount'],df['Profit'])
plt.xlabel('Discount')
plt.ylabel('profit')
plt.show()


# In[69]:


import matplotlib.pyplot as plt
discount_group = df.groupby("Discount")[["Profit"]].sum()

# Plotting
ax = discount_group.plot(kind="bar", figsize=(10, 5), legend=False)
ax.set_title("Total Profit by Discount")
ax.set_ylabel("Total Profit")
ax.set_xlabel("Discount Rate")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[70]:


## The best salles


# In[71]:


avg_profit_margin_by_catagory=df.groupby('Category')['Profit'].mean()
st.write(avg_profit_margin_by_catagory)


# In[72]:


##customers


# In[73]:


df.head()


# In[74]:


df['Ship Mode'].value_counts()


# In[75]:


pivot_table=pd.pivot_table(df, index='Segment', columns='Ship Mode', values='Sales', aggfunc='sum')


# In[76]:


pivot_table.plot(kind='bar',stacked=False)
plt.show()


# In[78]:


pivot_table


# Machine learning

# In[81]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# In[98]:


model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

features = ['Sales', 'Quantity', 'Discount']
target = 'Profit'

X = df[features]
y = df[target]


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[102]:


df = df[(df["Discount"] <= 0.8) & (df["Profit"] > -5000)]


# In[103]:


model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)


# In[104]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("MSE:", mse)
st.write("R² Score:", r2)


# In[21]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# In[22]:


df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Weekday"] = df["Order Date"].dt.dayofweek

target = "Profit"
features = ['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Region', 'Segment']
X = df[features]
y = df[target]


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


numeric_features = ['Sales', 'Quantity', 'Discount']
categorical_features = ['Category', 'Sub-Category', 'Region', 'Segment']


# In[25]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# In[26]:


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}


# In[27]:


results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MSE": mse, "R2": r2})
    results_df = pd.DataFrame(results)
best_model_row = results_df.loc[results_df["R2"].idxmax()]

st.write(" Best Estimator:")
st.write(f" Model: {best_model_row['Model']}")
st.write(f" R² Score: {best_model_row['R2']:.4f}")
st.write(f" MSE: {best_model_row['MSE']:.2f}")



# In[116]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(results_df["Model"], results_df["MSE"], color="salmon")
plt.title("Mean Squared Error (MSE)")
plt.xlabel("MSE")
plt.subplot(1, 2, 2)
plt.barh(results_df["Model"], results_df["R2"], color="skyblue")
plt.title("R² Score")
plt.xlabel("R²")

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib  # لحفظ النموذج وتصديره

# تحميل النموذج المدرب
model = joblib.load("best_model.pkl")

st.title("🔮 Profit Prediction App")

# إدخال المستخدم
sales = st.number_input("Sales", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
discount = st.slider("Discount", min_value=0.0, max_value=0.9, step=0.05)
category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
sub_category = st.selectbox("Sub-Category", ["Binders", "Phones", "Chairs", "Paper"])  # مثلاً
region = st.selectbox("Region", ["East", "West", "South", "Central"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

# تجهيز البيانات
input_data = pd.DataFrame({
    "Sales": [sales],
    "Quantity": [quantity],
    "Discount": [discount],
    "Category": [category],
    "Sub-Category": [sub_category],
    "Region": [region],
    "Segment": [segment]
})

# التنبؤ
if st.button("Predict Profit"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Expected Profit: {prediction:.2f} USD")


# In[32]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# تحميل البيانات
df = pd.read_csv("sample.csv", encoding="ISO-8859-1")

# الخصائص
features = ['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Region', 'Segment']
target = 'Profit'

X = df[features]
y = df[target]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# المعالجة
numeric_features = ['Sales', 'Quantity', 'Discount']
categorical_features = ['Category', 'Sub-Category', 'Region', 'Segment']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# النموذج داخل pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# تدريب النموذج
pipeline.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(pipeline, "best_model.pkl")  # ← هذا ما ينشئ الملف المطلوب

import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Profit Prediction", layout="centered")
st.title("🔮 Profit Prediction App (sample.csv)")

st.write("أدخل بيانات البيع لتوقع الربح المتوقع 👇")

# إدخال المستخدم
sales = st.number_input("Sales", min_value=0.0, value=100.0)
quantity = st.number_input("Quantity", min_value=1, value=1)
discount = st.slider("Discount", min_value=0.0, max_value=0.9, step=0.05)

category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
sub_category = st.selectbox("Sub-Category", ["Binders", "Paper", "Phones", "Chairs", "Accessories"])
region = st.selectbox("Region", ["East", "West", "Central", "South"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

# تجهيز البيانات
input_df = pd.DataFrame({
    "Sales": [sales],
    "Quantity": [quantity],
    "Discount": [discount],
    "Category": [category],
    "Sub-Category": [sub_category],
    "Region": [region],
    "Segment": [segment]
})

# التنبؤ
if st.button("🔍 Predict Profit"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Expected Profit: {prediction:.2f} USD")


# In[ ]:




