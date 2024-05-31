import os
import random

import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("model/nlp_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😱", "joy": "😆",
                       "neutral": "😐", "sadness": "😭", "shame": "😳", "surprise": "😮"}


def get_movies(emotion):
    movies_path = os.listdir('movies/' + emotion)
    movies_title = [os.path.splitext(movie)[0] for movie in movies_path if not movie.startswith('.')]
    recommend_movies = random.sample(movies_title, 4)
    return recommend_movies


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Movie Recommendation System")
    st.subheader("based on emotions with NLP")

    with st.form(key='my_form'):
        raw_text = st.text_area("Please enter a sentence that shows how you feel")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.info("Input Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}: {}".format(prediction, emoji_icon))
            st.write("Confidence: {}".format(np.max(probability)))

            st.warning("Prediction Probability")
            # st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            st.write(proba_df)
            # st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

        with col2:
            st.error("Let's watch these movies today!")

            img1, img2 = st.columns(2)
            base_dir = 'movies/' + prediction + '/'

            movies_list = get_movies(prediction)

            with img1:
                st.image(base_dir + movies_list[0] + '.jpg', caption=movies_list[0])
                st.image(base_dir + movies_list[1] + '.jpg', caption=movies_list[1])

            with img2:
                st.image(base_dir + movies_list[2] + '.jpg', caption=movies_list[2])
                st.image(base_dir + movies_list[3] + '.jpg', caption=movies_list[3])


if __name__ == '__main__':
    main()
