import streamlit as st
import pandas as pd

st.set_page_config(page_title="تحليل البيانات", layout="wide")
st.title("📊 تحليل ملف CSV باستخدام Streamlit")

# رفع ملف CSV من المستخدم
uploaded_file = st.file_uploader("📤 قم برفع ملف CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("👀 عرض البيانات")
        st.dataframe(df)

        st.subheader("📈 ملخص إحصائي")
        st.write(df.describe())

        # عرض رسم بياني إذا كان هناك أعمدة رقمية
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            st.subheader("📉 رسم بياني لعمود رقمي")
            selected_col = st.selectbox("اختر عمودًا", numeric_cols)
            st.line_chart(df[selected_col])
        else:
            st.info("لا توجد أعمدة رقمية لعرض رسم بياني.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
else:
    st.info("يرجى رفع ملف CSV للبدء.")
