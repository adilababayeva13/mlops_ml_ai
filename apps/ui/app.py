import os
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="What to Eat?", page_icon="🍜", layout="centered")

API_DEFAULT = os.getenv("API_URL", "http://what-to-eat-api:8000")
API = st.sidebar.text_input("API URL", API_DEFAULT)

st.title("🍜 What to Eat?")

tab1, tab2 = st.tabs(["Login", "Register"])

with tab2:
    st.subheader("Register")
    email = st.text_input("Email", key="r_email")
    password = st.text_input("Password", type="password", key="r_pass")
    if st.button("Create account"):
        r = requests.post(f"{API}/auth/register", json={"email": email, "password": password}, timeout=10)
        st.write(r.status_code)
        st.json(r.json())

with tab1:
    st.subheader("Login")
    email = st.text_input("Email", key="l_email")
    password = st.text_input("Password", type="password", key="l_pass")
    if st.button("Login"):
        r = requests.post(f"{API}/auth/login", json={"email": email, "password": password}, timeout=10)
        if r.status_code == 200:
            st.session_state["token"] = r.json()["access_token"]
            st.success("Logged in")
        else:
            st.error(r.json().get("detail", "Login failed"))

if "token" in st.session_state:
    st.header("🍽 Meal Recommendation (ML)")
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}

    q = requests.get(f"{API}/quiz/questions", headers=headers, timeout=10).json()

    answers = {}
    for item in q:
        answers[item["id"]] = st.radio(
            item["question"],
            options=[o["value"] for o in item["options"]],
            format_func=lambda x: next(o["label"] for o in item["options"] if o["value"] == x),
        )

    st.divider()

    if st.button("🔍 Recommend meals"):
        r = requests.post(f"{API}/quiz/submit", json=answers, headers=headers, timeout=30)

        if r.status_code != 200:
            st.error(r.text)
        else:
            data = r.json()
            recommended = data["result"]["recommended_meal"]

            st.subheader("🍽 Recommended meal (ML)")
            st.success(recommended)
            st.caption(f"Model version: `{data['model_version']}`")
            st.caption(f"Session id: `{data['session_id']}`")

            st.subheader("Your answers")
            st.json(answers)

            st.divider()
            st.subheader("Feedback (data flywheel)")

            accepted = st.radio("Was this recommendation good?", ["✅ Yes", "❌ No"], horizontal=True, key="accepted_radio")

            chosen_meal = recommended
            if accepted == "❌ No":
                chosen_meal = st.selectbox(
                    "Pick what you actually want instead:",
                    ["Vegan Buddha Bowl", "Spicy Chicken Bowl", "Keto Salmon Salad", "Margherita Pizza", "Beef Burrito"],
                    key="chosen_meal_select",
                )

            if st.button("Send feedback"):
                fb = {"session_id": data["session_id"], "chosen_meal": chosen_meal, "accepted": (accepted == "✅ Yes")}
                rr = requests.post(f"{API}/quiz/feedback", json=fb, headers=headers, timeout=10)
                if rr.status_code == 200:
                    st.success("Feedback saved ✅")
                else:
                    st.error(rr.text)
