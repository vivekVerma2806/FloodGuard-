"""FloodGuard AI - Premium Vintage Editorial Design System & Theme.

Visual Direction:
- 1960s scientific research journal & vintage cartographic atlas
- Minimal Swiss typographic discipline
- Monochromatic palette with warm ivory background (#F5F3ED), deep black (#171717),
  muted charcoal (#666666), hairline borders (#D6D3CC)
- Typography: Playfair Display / Libre Baskerville (headings), Inter (body),
  IBM Plex Mono (metrics, timestamps, scientific notation)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------------------------------------------------------
# Color Palette Constants
# -----------------------------------------------------------------------------
BG_COLOR = "#F5F3ED"
CARD_BG = "#FFFFFF"
CARD_BG_MUTED = "#FAF8F5"
TEXT_PRIMARY = "#171717"
TEXT_MUTED = "#666666"
TEXT_CAPTION = "#8C8880"
BORDER_COLOR = "#D6D3CC"
BORDER_DARK = "#171717"
ACCENT_BLACK = "#171717"

# Muted archival risk tones for cartographic clarity
RISK_PALETTE = {
    "Low": "#2E6930",        # Deep forest sage
    "Moderate": "#B45309",   # Archival ochre / amber
    "High": "#B91C1C",       # Deep brick red
    "Critical": "#6B1111",   # Dark oxblood / mahogany
}

# -----------------------------------------------------------------------------
# Custom Vector SVG Brand Mark
# -----------------------------------------------------------------------------
# Concept: Geodetic survey benchmark symbol intersected with concentric river
# elevation contour lines and hydrometric elevation gauge gradations.
BRAND_LOGO_SVG = """
<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Outer hairline boundary -->
  <rect x="1.5" y="1.5" width="45" height="45" stroke="#171717" stroke-width="1.2"/>
  <!-- Corner survey fiducials -->
  <line x1="1.5" y1="8" x2="6" y2="8" stroke="#171717" stroke-width="1"/>
  <line x1="8" y1="1.5" x2="8" y2="6" stroke="#171717" stroke-width="1"/>
  <line x1="42" y1="8" x2="46.5" y2="8" stroke="#171717" stroke-width="1"/>
  <line x1="40" y1="1.5" x2="40" y2="6" stroke="#171717" stroke-width="1"/>
  <line x1="1.5" y1="40" x2="6" y2="40" stroke="#171717" stroke-width="1"/>
  <line x1="8" y1="42" x2="8" y2="46.5" stroke="#171717" stroke-width="1"/>
  <line x1="42" y1="40" x2="46.5" y2="40" stroke="#171717" stroke-width="1"/>
  <line x1="40" y1="42" x2="40" y2="46.5" stroke="#171717" stroke-width="1"/>
  <!-- Topographic river elevation contour lines -->
  <path d="M 6 36 C 14 36, 18 30, 24 30 C 30 30, 34 34, 42 34" stroke="#171717" stroke-width="1.2" fill="none"/>
  <path d="M 6 27 C 13 27, 17 21, 24 21 C 31 21, 35 25, 42 25" stroke="#171717" stroke-width="1.5" fill="none"/>
  <path d="M 6 18 C 12 18, 16 12, 24 12 C 32 12, 36 16, 42 16" stroke="#171717" stroke-width="1.2" fill="none"/>
  <!-- Hydrometric gauge crosshair indicator -->
  <circle cx="24" cy="21" r="2.5" fill="#171717"/>
  <line x1="24" y1="6" x2="24" y2="10" stroke="#171717" stroke-width="1"/>
  <line x1="24" y1="32" x2="24" y2="42" stroke="#171717" stroke-width="1" stroke-dasharray="2 2"/>
</svg>
"""

BRAND_HEADER_HTML = f"""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid {BORDER_COLOR};">
  <div>{BRAND_LOGO_SVG}</div>
  <div>
    <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.35rem; font-weight: 700; letter-spacing: 0.18em; color: {TEXT_PRIMARY}; line-height: 1.1;">FLOODGUARD</div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: {TEXT_MUTED}; margin-top: 3px;">SPATIOTEMPORAL RISK INTELLIGENCE</div>
  </div>
</div>
"""

# -----------------------------------------------------------------------------
# Global Editorial CSS Injection
# -----------------------------------------------------------------------------
EDITORIAL_CSS = f"""
<style>
/* Import Classical Editorial & Scientific Monospace Fonts */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500;600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&display=swap');

/* Global Page Background & Base Text */
.stApp {{
    background-color: {BG_COLOR};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Typography System */
h1, h2, h3 {{
    font-family: 'Playfair Display', Georgia, serif !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}}

h1 {{
    font-size: 2.3rem !important;
    line-height: 1.15 !important;
    margin-bottom: 0.25rem !important;
}}

h2 {{
    font-size: 1.6rem !important;
    line-height: 1.25 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}}

h3 {{
    font-size: 1.2rem !important;
    line-height: 1.3 !important;
    margin-top: 1rem !important;
}}

p, span, label, div {{
    font-family: 'Inter', sans-serif;
    color: {TEXT_PRIMARY};
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: #ECEAE2 !important;
    border-right: 1px solid {BORDER_COLOR} !important;
    padding-top: 1.5rem;
}}

section[data-testid="stSidebar"] div.stRadio label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: 0.04em;
}}

/* Streamlit Navigation Sidebar Links */
section[data-testid="stSidebar"] ul {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{
    border-radius: 0px !important;
    border-left: 2px solid transparent !important;
    padding: 6px 12px !important;
    transition: all 0.15s ease-in-out;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{
    background-color: rgba(23, 23, 23, 0.05) !important;
    border-left: 2px solid {BORDER_COLOR} !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
    background-color: rgba(23, 23, 23, 0.08) !important;
    border-left: 2px solid {BORDER_DARK} !important;
    font-weight: 600 !important;
}}

/* Clean Hairline Dividers */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER_COLOR} !important;
    margin: 1.75rem 0 !important;
}}

/* Editorial Buttons - Minimal Dark Inset */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 1px !important;
    border: 1px solid {BORDER_DARK} !important;
    background-color: {BORDER_DARK} !important;
    color: {BG_COLOR} !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: none !important;
    transition: all 0.15s ease;
}}

.stButton > button:hover {{
    background-color: #383838 !important;
    border-color: #383838 !important;
    color: #FFFFFF !important;
}}

.stButton > button:active {{
    background-color: #000000 !important;
}}

/* Secondary / Download Buttons */
.stDownloadButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 1px !important;
    border: 1px solid {BORDER_DARK} !important;
    background-color: transparent !important;
    color: {BORDER_DARK} !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: none !important;
    transition: all 0.15s ease;
}}

.stDownloadButton > button:hover {{
    background-color: {BORDER_DARK} !important;
    color: {BG_COLOR} !important;
}}

/* Selectbox, Input, and Radio Refinement */
div[data-baseweb="select"] > div {{
    border-radius: 1px !important;
    border-color: {BORDER_COLOR} !important;
    background-color: {CARD_BG} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}}

div[data-baseweb="input"] > div {{
    border-radius: 1px !important;
    border-color: {BORDER_COLOR} !important;
    background-color: {CARD_BG} !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

/* Metric Cards & Data Tables */
div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
}}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {BORDER_COLOR} !important;
    gap: 1.5rem !important;
}}

.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-radius: 0px !important;
    padding: 0.5rem 0 !important;
    color: {TEXT_MUTED} !important;
}}

.stTabs [aria-selected="true"] {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {BORDER_DARK} !important;
}}

/* Expanders */
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 1px !important;
    background-color: {CARD_BG} !important;
}}

div[data-testid="stExpander"] details summary span {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    color: {TEXT_PRIMARY} !important;
}}

/* DataFrames */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_COLOR} !important;
    background-color: {CARD_BG} !important;
}}
</style>
"""


def apply_theme():
    """Inject global CSS rules into the current Streamlit page."""
    st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


def render_sidebar():
    """Render the standard institutional brand sidebar."""
    with st.sidebar:
        st.markdown(BRAND_HEADER_HTML, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: {TEXT_MUTED}; line-height: 1.6; padding: 0.5rem 0; border-top: 1px solid {BORDER_COLOR}; border-bottom: 1px solid {BORDER_COLOR}; margin-bottom: 1rem;">
                <div>BULLETIN // HYDRO-RISK</div>
                <div>STANDARD: SPATIOTEMPORAL V3</div>
                <div>REFERENCE: CWC-BASIN-5</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def editorial_header(title: str, kicker: str = "HYDROLOGICAL RESEARCH ARCHIVE // SYSTEM REPORT", subtitle: str = None):
    """Render a vintage editorial article/dashboard header."""
    subtitle_html = (
        f'<div style="font-family: \'Libre Baskerville\', Georgia, serif; font-style: italic; font-size: 0.95rem; color: {TEXT_MUTED}; margin-top: 6px; max-width: 820px; line-height: 1.5;">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid {BORDER_COLOR};">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase; color: {TEXT_CAPTION}; margin-bottom: 4px;">{kicker}</div>
            <h1 style="margin: 0; padding: 0;">{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value, caption: str = None, delta=None):
    """Render a refined minimalist editorial metric card."""
    delta_html = ""
    if delta is not None:
        delta_str = str(delta)
        delta_color = TEXT_MUTED
        if delta_str.startswith("-"):
            delta_color = "#2E6930"  # reduction in risk/error
        elif delta_str.startswith("+"):
            delta_color = "#B45309"
        delta_html = f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.72rem; color: {delta_color}; margin-top: 4px;">{delta_str}</div>'

    caption_html = (
        f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.74rem; color: {TEXT_MUTED}; margin-top: 4px; line-height: 1.3;">{caption}</div>'
        if caption
        else ""
    )

    st.markdown(
        f"""
        <div style="background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; padding: 1.1rem 1.25rem; border-radius: 1px; min-height: 110px; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: {TEXT_MUTED};">{title}</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem; font-weight: 600; color: {TEXT_PRIMARY}; letter-spacing: -0.02em; margin-top: 4px; line-height: 1.1;">{value}</div>
            {delta_html}
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def editorial_callout(text: str, label: str = "RESEARCH NOTE", level: str = "info"):
    """Render an archival marginal note or technical warning."""
    indicator_color = BORDER_DARK if level == "info" else "#B45309"
    st.markdown(
        f"""
        <div style="background-color: {CARD_BG_MUTED}; border-left: 3px solid {indicator_color}; border-top: 1px solid {BORDER_COLOR}; border-right: 1px solid {BORDER_COLOR}; border-bottom: 1px solid {BORDER_COLOR}; padding: 0.85rem 1.15rem; margin: 1rem 0; border-radius: 1px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; letter-spacing: 0.14em; text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 3px;">// {label}</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.86rem; color: {TEXT_PRIMARY}; line-height: 1.45;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Vintage Plotly Editorial Layout Helper
# -----------------------------------------------------------------------------
def style_plotly_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    """Format Plotly charts into a vintage scientific publication style."""
    fig.update_layout(
        template="none",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=40, r=25, t=45, b=40),
        font=dict(
            family="IBM Plex Mono, monospace",
            size=11,
            color=TEXT_PRIMARY,
        ),
        title=dict(
            font=dict(
                family="Playfair Display, Georgia, serif",
                size=15,
                color=TEXT_PRIMARY,
            ),
            x=0.01,
            y=0.96,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(size=10, family="IBM Plex Mono, monospace"),
        ),
        xaxis=dict(
            gridcolor="#ECE8DF",
            linecolor=BORDER_COLOR,
            linewidth=1,
            tickfont=dict(size=10, family="IBM Plex Mono, monospace"),
            title_font=dict(size=11, family="IBM Plex Mono, monospace"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#ECE8DF",
            linecolor=BORDER_COLOR,
            linewidth=1,
            tickfont=dict(size=10, family="IBM Plex Mono, monospace"),
            title_font=dict(size=11, family="IBM Plex Mono, monospace"),
            zeroline=False,
        ),
    )
    return fig
