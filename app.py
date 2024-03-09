import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

st.title('Blog Gender Classifier')
st.text('')
st.subheader('Please enter the blog post in the below area -', divider='rainbow')
blog = st.text_area("Blog Text")
st.write(f'You wrote {len(blog)} characters.')

if st.button("Predict", type="primary"):
    if blog != "":
        #model = pickle.load(open('Blog_navbaise_model.pkl', 'rb'))
        data = pd.read_excel('data/BLOG GENDER BALANCED.xlsx')
        data.dropna(subset=["BLOG"],inplace=True)
        data1 = pd.DataFrame({
            "BLOG": [blog],
            "GENDER": ["M"],
        }, index=[len(data)+1])

        data = pd.concat([data,data1])

        X = data["BLOG"]
        y = data["GENDER"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        #vectorizer = CountVectorizer(stop_words='english')
        vectorizer = TfidfVectorizer(stop_words='english')
        ctmTr = vectorizer.fit_transform(X_train)
        X_test_dtm = vectorizer.transform(X_test)

        clf = MultinomialNB()
        clf.fit(ctmTr, y_train)

        y_pred = clf.predict(X_test_dtm)
        st.dataframe(
            pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True)
            ).transpose()
        )
        prediction = pd.DataFrame(y_pred.T, columns=['Pred'])
        if str(prediction['Pred'].tail(1).values[0]) == 'M':
            st.write('This Blog written by a Male author!')
        else:
            st.write('This Blog written by a Female author!')
    else:
        st.error('Please enter blog text!')
    





