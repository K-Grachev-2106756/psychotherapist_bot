import os
import gc
import json
from time import time
from enum import Enum

import torch
import streamlit as st

from inference.chat import Chat
from inference.llm import LLM
from inference.classifier import Classifier
from config import (
    base_model_name,
    cbt_lora_path, 
    emo_lora_path,
    max_tokens,
    class_mapping,
    clf_model_name,
    clf_model_path,
    default_streamlit_model,
)


class AvailableModels(Enum):
    BASE = "Base"
    CLF = "Base with Emotion Classifier"
    CBT = "Tuned for CBT"
    EMO = "Tuned for CBT with Emotional Intelligence"


AVAILABLE_MODELS = [model_mode.value for model_mode in AvailableModels]


# @st.cache_resource
def load_model():
    model_mode = AvailableModels(st.session_state.model_name)
    
    base_model, tokenizer = LLM.init(model_name=base_model_name)
    clf_params = {
        "model_name": clf_model_name,
        "model_path": clf_model_path,
        "class_mapping": class_mapping,
    }

    match model_mode:
        case AvailableModels.BASE | AvailableModels.CLF:
            clf = None if model_mode == AvailableModels.BASE else Classifier(**clf_params)
            params = {
                "llm_model": base_model,
                "classifier": clf,
            } 
        case AvailableModels.CBT:
            cbt_model = LLM.load_lora(
                base_model=base_model, 
                model_path=cbt_lora_path,
            )
            params = {
                "llm_model": cbt_model,
                "classifier": None,
            } 
        case AvailableModels.EMO:
            emo_model = LLM.load_lora(
                base_model=base_model, 
                model_path=emo_lora_path,
            )
            params = {
                "llm_model": emo_model,
                "classifier": Classifier(**clf_params),
            } 
        case _:
            raise ValueError(model_mode)
        
    params.update({
        "llm_tokenizer": tokenizer,
        "max_context_tokens": max_tokens,
    })

    return params
    

def make_json_id():
    model_name = {
        "Base": "BASE",
        "Base with Emotion Classifier": "CLF",
        "Tuned for CBT": "CBT",
        "Tuned for CBT with Emotional Intelligence": "EMO",
    }.get(st.session_state.model_name)

    st.session_state.id = model_name + "-" + str(time()).replace(".", "-")


def save_json():
    if len(st.session_state.chat_history_display):
        file = os.path.join("ratings", st.session_state.id + ".json")
        with open(file, "w", encoding="utf-8") as f:
            data = [
                {"role": role, "text": message, "rating": st.session_state.ratings.get(idx, {})}
                for idx, (role, message) in enumerate(st.session_state.chat_history_display)
            ] 
            if "global" in st.session_state.ratings:
                data[0]["global"] = st.session_state.ratings["global"]

            json.dump(data, f, ensure_ascii=False, indent=4)


def model_switch_cancel_func():
    st.session_state.selected_model_name = st.session_state.model_name
    st.session_state.pending_model = None
    st.session_state.show_model_dialog = False
    st.session_state.pressed_model_dialog = True
    st.rerun()


def reset_chat_func():
    if st.session_state.ratings:
        save_json()
        make_json_id()

    st.session_state.chat.reset()
    st.session_state.ratings = {}
    st.session_state.chat_history_display = []
    st.session_state.show_reset_dialog = False
    st.session_state.pressed_reset_dialog = True


@st.dialog("Confirmation dialog")
def confirm_model_switch():
    st.session_state.shown_model_dialog = True
    st.write("Are you sure you want to switch models? All data will be lost.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm", key="model_switch_confirmation"):
            new_model_name = st.session_state.pending_model
            st.session_state.model_name = new_model_name
            st.session_state.selected_model_name = new_model_name

            save_json()
            make_json_id()
            st.session_state.chat_history_display = []
            st.session_state.ratings = {}

            del st.session_state.chat
            torch.cuda.empty_cache()
            gc.collect()
            st.session_state.chat = Chat(**load_model())
            
            st.session_state.pending_model = None
            st.session_state.show_model_dialog = False
            st.session_state.pressed_model_dialog = True
            st.rerun()
    with col2:
        if st.button("❌ Cancel", key="model_switch_cancel"):
            model_switch_cancel_func()


@st.dialog("Evaluate the model's response")
def rate_response(index):
    st.write("Evaluate the response of AI-assistant from 1 to 10 for each criterion:")

    criteria = [
        "Correctness/adequacy/relevance/completeness",
        "How empathetic is the response",
        "Usefulness of the tips",
        "The overall feeling of the dialogue",
    ]

    inputs = {}
    for c in criteria:
        inputs[c] = st.number_input(c, min_value=1, max_value=10, step=1, key=f"rate_{c}_{index}")

    if st.button("📨 Send", key="send_response_rate"):
        st.session_state.ratings[index] = inputs
        save_json()
        st.rerun()


@st.dialog("Evaluate the session before resetting")
def rate_session():
    st.session_state.shown_reset_dialog = True
    if len(st.session_state.chat_history_display):
        st.write("Please rate the session according to the following criteria:")
        score = st.number_input("How useful was the whole session for you?", min_value=1, max_value=10, step=1, key="global_rating")
        if st.button("📨 Send and clear", key="global_session_rate"):
            st.session_state.ratings["global"] = score
            reset_chat_func()
            st.rerun()
    else:
        reset_chat_func()
        st.rerun()




# ----- Инициализация session_state -----
if not st.session_state.get("initialized"):
    st.session_state.model_name = default_streamlit_model
    st.session_state.selected_model_name = default_streamlit_model
    
    st.session_state.chat = Chat(**load_model())
    st.session_state.chat_history_display = []
    st.session_state.chat_history_display = []
    st.session_state.ratings = {}

    st.session_state.pending_model = None
    st.session_state.show_model_dialog = False
    st.session_state.shown_model_dialog = False
    st.session_state.pressed_model_dialog = False
    st.session_state.show_reset_dialog = False
    st.session_state.shown_reset_dialog = False
    st.session_state.pressed_reset_dialog = False

    os.makedirs("ratings", exist_ok=True)
    make_json_id()

    st.session_state.initialized = True


# ----- Заголовок -----
st.title("🧠 CBT AI-Chatbot")


# ----- История сообщений -----
for idx, (role, message) in enumerate(st.session_state.chat_history_display):
    st.chat_message(role).write(message)

    if (role == "assistant") and (idx not in st.session_state.ratings):
        if st.button("Evaluate the answer", key=f"rate_button_{idx}"):
            rate_response(idx)


# ----- Ввод пользователя -----
user_input = st.chat_input("Share what's on your mind 😌")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history_display.append(("user", user_input))

    with st.spinner("The therapist thinks hard..."):
        response = st.session_state.chat.ask(user_input)
        print(st.session_state.chat.tokenizer.decode(st.session_state.chat.history.get_context()[0], skip_special_tokens=True))  ##########################

    st.chat_message("assistant").write(response)
    st.session_state.chat_history_display.append(("assistant", response))
    idx = len(st.session_state.chat_history_display) + 1
    if st.button("Evaluate the answer", key=f"rate_button_{idx}"):
        rate_response(idx)


# ----- Диалоги -----
if st.session_state.show_model_dialog:
    if (st.session_state.shown_model_dialog) and (st.session_state.pressed_model_dialog):
        st.session_state.shown_model_dialog = False
        st.session_state.pressed_model_dialog = False
        confirm_model_switch()
    elif (st.session_state.shown_model_dialog) and (not st.session_state.pressed_model_dialog):
        model_switch_cancel_func()
    else:
        confirm_model_switch()


if st.session_state.show_reset_dialog:
    if (st.session_state.shown_reset_dialog) and (st.session_state.pressed_reset_dialog):
        st.session_state.shown_reset_dialog = False
        st.session_state.pressed_reset_dialog = False
        rate_session()
    elif (st.session_state.shown_reset_dialog) and (not st.session_state.pressed_reset_dialog):
        st.session_state.show_reset_dialog = False
        st.session_state.pressed_reset_dialog = True
    else:
        rate_session()


# ----- Боковая панель -----
with st.sidebar:
    st.header("Settings")

    selected_name = st.radio(
        "Select a model",
        AVAILABLE_MODELS,
        key="selected_model_name",
    )

    if (selected_name != st.session_state.model_name) and (not st.session_state.show_model_dialog):
        st.session_state.pending_model = selected_name
        st.session_state.show_model_dialog = True
        st.rerun()

    if st.button("🗑️ Delete the dialog", key="clear_chat") and (not st.session_state.show_reset_dialog):
        st.session_state.show_reset_dialog = True
        st.rerun()
