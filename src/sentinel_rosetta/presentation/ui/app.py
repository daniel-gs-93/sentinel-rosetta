"""Resource-aware: Streamlit page that calls the translate API."""

import os

import requests
import streamlit as st

_API_BASE_URL = os.environ.get("SENTINEL_API_BASE_URL", "http://localhost:8000")
_LANGUAGE_OPTIONS = ["en", "zh"]


def call_translate_api(
    text: str, source_lang: str, target_lang: str
) -> requests.Response:
    return requests.post(
        f"{_API_BASE_URL}/api/v1/translate",
        json={
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
        timeout=30,
    )


st.title("Sentinel Rosetta")

text = st.text_area("Text")
source_lang = st.selectbox("Source language", _LANGUAGE_OPTIONS)
target_lang = st.selectbox("Target language", _LANGUAGE_OPTIONS, index=1)

if st.button("Translate"):
    try:
        response = call_translate_api(text, source_lang, target_lang)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the translation API: {e}")
    else:
        try:
            body = response.json()
        except ValueError:
            st.error(
                f"Translation API returned a non-JSON response: {response.text}"
            )
        else:
            if not response.ok:
                st.error(body.get("detail", response.text))
            elif "target_text" not in body:
                st.error(
                    f"Translation API response is missing target_text: {body}"
                )
            else:
                st.success(body["target_text"])
