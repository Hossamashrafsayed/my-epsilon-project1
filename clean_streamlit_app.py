
import streamlit as st
import pandas as pd

# إعداد صفحة التطبيق
st.set_page_config(page_title="تحليل البيانات", layout="wide")

# عنوان رئيسي
st.title("📊 تطبيق Streamlit لتحليل البيانات")

# تحميل البيانات
@st.cache_data
def load_data():
    return pd.read_csv("sample.csv")

df = load_data()

# عرض البيانات
st.subheader("👀 عرض البيانات")
st.dataframe(df)

# عرض ملخص إحصائي
st.subheader("📈 ملخص إحصائي")
st.write(df.describe())

# اختيار عمود للرسم البياني (إن وجد)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numeric_cols:
    st.subheader("📉 رسم بياني لعمود معين")
    selected_col = st.selectbox("اختر عمودًا رقميًا", numeric_cols)
    st.line_chart(df[selected_col])
else:
    st.info("لا توجد أعمدة رقمية لعرض الرسوم البيانية.")
