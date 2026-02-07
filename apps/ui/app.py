import streamlit as st
import requests
import pandas as pd

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="What to Eat?",
    page_icon="🍜",
    layout="centered",
)

if "API_URL" not in st.session_state:
    st.session_state.API_URL = "http://34.80.4.247"

API = st.sidebar.text_input(
    "API URL",
    key="API_URL"
)

# API = 'http://34.80.4.247'

st.title("🍜 What to Eat?")

# =========================================================
# AUTH
# =========================================================
tab1, tab2 = st.tabs(["Login", "Register"])

with tab2:
    st.subheader("Register")
    email = st.text_input("Email", key="r_email")
    password = st.text_input("Password", type="password", key="r_pass")

    if st.button("Create account"):
        r = requests.post(
            f"{API}/auth/register",
            json={"email": email, "password": password},
            timeout=10,
        )
        st.write(r.status_code)
        st.json(r.json())

with tab1:
    st.subheader("Login")
    email = st.text_input("Email", key="l_email")
    password = st.text_input("Password", type="password", key="l_pass")

    if st.button("Login"):
        r = requests.post(
            f"{API}/auth/login",
            json={"email": email, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            st.session_state.token = r.json()["access_token"]
            st.success("Logged in")
        else:
            st.error(r.json().get("detail", "Login failed"))

# =========================================================
# REQUIRE AUTH
# =========================================================
if "token" not in st.session_state:
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

st.header("🍽 Meal Recommendation (ML)")

# =========================================================
# LOAD QUESTIONS
# =========================================================
resp = requests.get(f"{API}/quiz/questions", headers=headers)
if resp.status_code != 200:
    st.error("Failed to load quiz questions")
    st.stop()

questions = resp.json()

# =========================================================
# QUIZ
# =========================================================
answers = {}
for q in questions:
    answers[q["id"]] = st.radio(
        q["question"],
        options=[o["value"] for o in q["options"]],
        format_func=lambda x: next(
            o["label"] for o in q["options"] if o["value"] == x
        ),
        key=f"q_{q['id']}",
    )

st.divider()

# =========================================================
# SUBMIT QUIZ
# =========================================================
if st.button("🔍 Recommend meals"):
    r = requests.post(
        f"{API}/quiz/submit",
        json=answers,
        headers=headers,
        timeout=30,
    )

    if r.status_code != 200:
        st.error(r.text)
    else:
        data = r.json()

        # Persist prediction state
        st.session_state.session_id = data["session_id"]
        st.session_state.model_version = data["model_version"]
        st.session_state.result = data["result"]

        st.session_state.recommended_meal = data["result"]["recommended_meal"]
        st.session_state.top_k = data["result"]["top_k"]

        # Reset feedback state
        st.session_state.accepted = "✅ Yes"
        st.session_state.chosen_meal = st.session_state.recommended_meal

# =========================================================
# SHOW RESULT (PERSISTENT)
# =========================================================
if "recommended_meal" in st.session_state:
    st.subheader("🍽 Recommended meal")
    st.success(st.session_state.recommended_meal)

    st.caption(f"Model version: `{st.session_state.model_version}`")
    st.caption(f"Session id: `{st.session_state.session_id}`")

    st.subheader("Top-K candidates (with confidence)")
    df_topk = pd.DataFrame(st.session_state.top_k)
    st.dataframe(df_topk, use_container_width=True)
    st.bar_chart(df_topk.set_index("meal")["prob"])

    st.subheader("Your answers (model inputs)")
    st.json(answers)

# =========================================================
# FEEDBACK (DATA FLYWHEEL)
# =========================================================
if "session_id" in st.session_state:
    st.divider()
    st.subheader("Feedback (data flywheel)")

    accepted = st.radio(
        "Was this recommendation good?",
        ["✅ Yes", "❌ No"],
        horizontal=True,
        key="accepted_radio",
    )
    st.session_state.accepted = accepted

    # Build selectable meals list
    all_meals = [
        "Vegan Buddha Bowl",
        "Spicy Chicken Bowl",
        "Keto Salmon Salad",
        "Margherita Pizza",
        "Beef Burrito",
    ]
    topk_meals = [x["meal"] for x in st.session_state.top_k]
    meal_options = list(dict.fromkeys(topk_meals + all_meals))

    if accepted == "❌ No":
        chosen_meal = st.selectbox(
            "Pick what you actually want instead:",
            meal_options,
            key="chosen_meal_select",
        )
        st.session_state.chosen_meal = chosen_meal
    else:
        st.session_state.chosen_meal = st.session_state.recommended_meal

    if st.button("Send feedback"):
        payload = {
            "session_id": st.session_state.session_id,
            "chosen_meal": st.session_state.chosen_meal,
            "accepted": st.session_state.accepted == "✅ Yes",
        }

        rr = requests.post(
            f"{API}/quiz/feedback",
            json=payload,
            headers=headers,
            timeout=10,
        )

        if rr.status_code == 200:
            st.success("Feedback saved ✅")
        else:
            st.error(rr.text)




st.divider()
st.header("🤖 Ask AI for meal recommendation")

if "token" in st.session_state:
    user_msg = st.text_area(
        "Describe what you want to eat",
        placeholder="I want something spicy, cheap, and quick to prepare",
    )

    if st.button("Ask AI"):
        r = requests.post(
            f"{API}/llm/recommend",
            json={"message": user_msg},
            headers=headers,
            timeout=30,
        )

        if r.status_code != 200:
            st.error(r.text)
        else:
            result = r.json()
            st.success(result["meal"])
            st.write(result["reason"])
            st.write("Tags:", result["tags"])
            st.progress(int(result["confidence"] * 100))
