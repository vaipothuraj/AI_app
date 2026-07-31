import streamlit as st
import requests

import os
from dotenv import load_dotenv
from openai import OpenAI

#

st.title("Welcome to MEDIA, our first AI model online.")







name= st.text_input("What is your name?" )
if st.button("Submit"):
    st.write(f"Hello, {name} Welcome to AI Level 2!")
    st.write(f"Hi {name}, your name has {len(name)} letters.")


#for a in name:
 #   vowels =0
  #  vowels = vowels + 1 if a == a or o or i or u
   # st.write(f"Your name has {vowels} vowels.")

with st.sidebar:
    st.header("Settings tab")
    with st.form("settings"):
        sources = st.multiselect("Select a few options",["My first app","My second app"])
        creativity = st.slider("Creativity", 0.0, 1.0, 0.5)
        saved = st.form_submit_button("Save")
    if saved:
        st.write(f"Saved sources; {sources} and creativity:{creativity}")


#st.text_input(typing box)
#st.selectbox(pick from menu)
#st.multiselect (gives list)
#st.slider (drags handle along line)
with st.chat_message("user"):
    st.write(f"Hello, I am {name}")
with st.chat_message("assistant"):
    st.write(f"I am MEDIA, welcome to my world.")


prompt = (f"You are a harsh schoolteacher who doesn't accept apologies or affection. Do not give opinions on religious subjects. Use simple words with no jargon. Question: {st.chat_input("Ask me something here:")}")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        if prompt == "Cat Fact":
            r = requests.get("https://catfact.ninja/fact")
            fact = r.json()["fact"]
            st.write(f"{fact}")
        else:
            load_dotenv()
            client = OpenAI(
                     base_url="https://api.groq.com/openai/v1",
                     api_key=os.environ.get("AI_TOKEN") or st.secrets["AI_TOKEN"],
            )
            r = client.chat.completions.create(
                 model="llama-3.3-70b-versatile",
                 messages=[{"role": "user", "content": prompt}],
            )

            st.write(r.choices[0].message.content)

left, right = st.columns(2)
left.write(f"Sources={len(sources)}")
right.write(f"Creativity={creativity}")


