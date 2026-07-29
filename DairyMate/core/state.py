"""
Keeping work alive across page switches.

Streamlit drops a widget's value once the widget stops being rendered, so
navigating from Teat analysis to Udder analysis and back throws away the
uploaded file and everything computed from it. Session state under our own
keys is not subject to that cleanup, so the image bytes and the last result
are mirrored there and rehydrated on the way back.

Results are keyed by a signature over the image and the settings that affect
the score, so changing a cut-off recomputes but merely revisiting the page
does not — which matters on the teat page, where a Grad-CAM pass is not free.
"""
from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st


def _key(slot: str, field: str) -> str:
    # Deliberately not a widget key — widget keys get garbage collected.
    return f"_dm::{slot}::{field}"


# ------------------------------------------------------------------ images
def remember_image(slot: str, uploaded) -> tuple[bytes | None, str | None]:
    """
    Store a fresh upload, or hand back the one from last time.

    Returns (bytes, filename), both None if nothing has been uploaded yet.
    """
    if uploaded is not None:
        data = uploaded.getvalue()
        if data:
            st.session_state[_key(slot, "bytes")] = data
            st.session_state[_key(slot, "name")] = uploaded.name
    return (
        st.session_state.get(_key(slot, "bytes")),
        st.session_state.get(_key(slot, "name")),
    )


def forget(slot: str) -> None:
    for field in ("bytes", "name", "sig", "result", "extra"):
        st.session_state.pop(_key(slot, field), None)


def is_restored(slot: str, uploaded) -> bool:
    """True when we are showing a remembered image rather than a new upload."""
    return uploaded is None and st.session_state.get(_key(slot, "bytes")) is not None


# ----------------------------------------------------------------- results
def signature(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            h.update(part)
        else:
            h.update(repr(part).encode())
        h.update(b"|")
    return h.hexdigest()


def cached_result(slot: str, sig: str):
    """The stored result if it was computed under this exact signature."""
    if st.session_state.get(_key(slot, "sig")) == sig:
        return st.session_state.get(_key(slot, "result"))
    return None


def store_result(slot: str, sig: str, result, extra: Any = None) -> None:
    st.session_state[_key(slot, "sig")] = sig
    st.session_state[_key(slot, "result")] = result
    if extra is not None:
        st.session_state[_key(slot, "extra")] = extra


def cached_extra(slot: str):
    return st.session_state.get(_key(slot, "extra"))
