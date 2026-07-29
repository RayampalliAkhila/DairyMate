"""Reusable interface pieces."""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import config, theme


# --------------------------------------------------------------- masthead
def masthead(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="dm-masthead">
          <div class="dm-eyebrow">{html.escape(eyebrow)}</div>
          <div class="dm-title">{html.escape(title)}</div>
          <div class="dm-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="dm-caption">{html.escape(text)}</div>', unsafe_allow_html=True)


def stat(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="dm-stat__note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="dm-stat">
          <div class="dm-stat__k">{html.escape(label)}</div>
          <div class="dm-stat__v">{html.escape(value)}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "warn") -> None:
    cls = {"warn": "dm-note", "info": "dm-note dm-note--info", "ok": "dm-note dm-note--ok"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- verdict
def verdict(label_key: str, probability: float, threshold: float) -> None:
    """Headline result. label_key is 'healthy' or 'mastitis'."""
    colour = theme.STATUS[label_key]
    text = config.CLASS_LABELS[label_key]
    margin = abs(probability - threshold)
    if margin < 0.08:
        text += " — borderline"
        colour = theme.CAUTION
    st.markdown(
        f"""
        <div class="dm-verdict" style="border-left-color:{colour};">
          <div class="dm-verdict__label" style="color:{colour};">{html.escape(text)}</div>
          <div class="dm-verdict__score">p(mastitis) {probability:.3f} &nbsp;·&nbsp; cut-off {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------- signature: decision strip
def decision_strip(probability: float, threshold: float, height: int = 74) -> None:
    """
    The one element this console is built around.

    A classifier score is meaningless without the cut-off it is compared to,
    and on the udder model that cut-off is both calibrated and fragile. So the
    strip draws the full 0-1 range, marks the calibrated cut-off as a hard
    notch, and puts the score on top of it — you read the decision and the
    distance from the decision in the same glance.
    """
    w, pad = 760, 14
    inner = w - 2 * pad
    x_p = pad + inner * max(0.0, min(1.0, probability))
    x_t = pad + inner * max(0.0, min(1.0, threshold))
    bar_y, bar_h = 26, 16

    svg = f"""
    <svg viewBox="0 0 {w} {height}" width="100%" height="{height}"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Score {probability:.3f} against cut-off {threshold:.2f}">
      <defs>
        <linearGradient id="dmgrad" x1="0" x2="1">
          <stop offset="0%"   stop-color="{theme.HEALTHY}" stop-opacity="0.85"/>
          <stop offset="50%"  stop-color="#C9D3D8"/>
          <stop offset="100%" stop-color="{theme.MASTITIS}" stop-opacity="0.9"/>
        </linearGradient>
      </defs>

      <rect x="{pad}" y="{bar_y}" width="{inner}" height="{bar_h}" rx="2" fill="url(#dmgrad)"/>
      <rect x="{pad}" y="{bar_y}" width="{inner}" height="{bar_h}" rx="2"
            fill="none" stroke="{theme.LINE}"/>

      <!-- calibrated cut-off -->
      <line x1="{x_t:.1f}" y1="{bar_y - 7}" x2="{x_t:.1f}" y2="{bar_y + bar_h + 7}"
            stroke="{theme.INK}" stroke-width="2" stroke-dasharray="3 2"/>
      <text x="{x_t:.1f}" y="{bar_y + bar_h + 20}" text-anchor="middle"
            font-family="IBM Plex Mono, monospace" font-size="10" fill="{theme.INK}">
        cut-off {threshold:.2f}
      </text>

      <!-- score -->
      <polygon points="{x_p:.1f},{bar_y - 3} {x_p - 6:.1f},{bar_y - 13} {x_p + 6:.1f},{bar_y - 13}"
               fill="{theme.INK}"/>
      <text x="{x_p:.1f}" y="{bar_y - 17}" text-anchor="middle"
            font-family="IBM Plex Mono, monospace" font-size="11"
            font-weight="600" fill="{theme.INK}">{probability:.3f}</text>

      <text x="{pad}" y="{bar_y + bar_h + 20}" font-family="IBM Plex Mono, monospace"
            font-size="10" fill="{theme.MUTED}">0.00 healthy</text>
      <text x="{w - pad}" y="{bar_y + bar_h + 20}" text-anchor="end"
            font-family="IBM Plex Mono, monospace" font-size="10"
            fill="{theme.MUTED}">mastitis 1.00</text>
    </svg>
    """
    # components.html rather than st.markdown: an iframe renders the SVG
    # verbatim, where markdown sanitising can silently drop parts of it.
    components.html(
        f'<div style="margin:0;padding:0;background:transparent;">{svg}</div>',
        height=height + 6,
    )


# --------------------------------------------------------------- sidebar
def sidebar_status(teat_root: Path | None, udder_root: Path | None) -> None:
    with st.sidebar:
        st.markdown("### Dairy Mate")
        st.caption("Mastitis screening console")
        st.divider()
        st.markdown("**Pipelines**")
        for name, root in (("Teat", teat_root), ("Udder", udder_root)):
            if root is None:
                st.markdown(f"✗ {name} — not found")
            else:
                st.markdown(f"✓ {name}")
                st.caption(str(root))
        st.divider()
        st.caption(
            "Screening aid only. Every flagged animal needs a hands-on check "
            "before treatment."
        )
        st.caption(f"v{config.APP_VERSION}")


def missing_pipeline(kind: str, folder: str) -> None:
    st.error(f"The {kind} pipeline folder was not found.")
    st.markdown(
        f"""
Put the `{folder}` folder next to this app, or start the app with the path set:

```bash
{kind.upper()}_PIPELINE_DIR=/full/path/to/{folder} streamlit run app.py
```

The app expects to find `{folder}/models/` and `{folder}/reports/` inside it.
"""
    )
