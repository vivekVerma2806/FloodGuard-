"""Unit tests for Markdown and editorial text rendering engine.

Ensures that bold, italic, code, strikethrough, links, lists, and headings
render into proper semantic HTML tags with zero leaking raw syntax (e.g. `**text**`).
"""

import pytest
from src.ui_theme import format_markdown_text, editorial_callout


class TestMarkdownRendering:
    """Test suite for format_markdown_text and editorial text transformation."""

    def test_bold_double_asterisk(self):
        text = "Optimal baseline regression algorithm: **Random Forest** (RMSE = 1.1568)."
        formatted = format_markdown_text(text)
        assert "<strong>Random Forest</strong>" in formatted
        assert "**Random Forest**" not in formatted

    def test_bold_double_underscore(self):
        text = "Performance audit: __Linear Regression__ selected."
        formatted = format_markdown_text(text)
        assert "<strong>Linear Regression</strong>" in formatted
        assert "__Linear Regression__" not in formatted

    def test_italic_single_asterisk(self):
        text = "This metric represents *provisional* telemetry."
        formatted = format_markdown_text(text)
        assert "<em>provisional</em>" in formatted
        assert "*provisional*" not in formatted

    def test_strikethrough(self):
        text = "Status: ~~deprecated~~ superseded by LSTM."
        formatted = format_markdown_text(text)
        assert "<del>deprecated</del>" in formatted
        assert "~~deprecated~~" not in formatted

    def test_inline_code(self):
        text = "Verify key in `st.session_state['lstm_model']`."
        formatted = format_markdown_text(text)
        assert "<code" in formatted
        assert "st.session_state['lstm_model']" in formatted
        assert "`st.session_state['lstm_model']`" not in formatted

    def test_markdown_links(self):
        text = "Reference: [Central Water Commission](https://cwc.gov.in) portal."
        formatted = format_markdown_text(text)
        assert '<a href="https://cwc.gov.in"' in formatted
        assert 'target="_blank"' in formatted
        assert "Central Water Commission</a>" in formatted

    def test_composite_markdown_formatting(self):
        text = "• Station **Godavari Basin - Bhadrachalam**: *Stage* at `6.45m` (~~5.80m~~)."
        formatted = format_markdown_text(text)
        assert "<strong>Godavari Basin - Bhadrachalam</strong>" in formatted
        assert "<em>Stage</em>" in formatted
        assert "<code" in formatted
        assert "<del>5.80m</del>" in formatted
        assert "**" not in formatted
        assert "~~" not in formatted

    def test_multiplication_not_converted_to_italics(self):
        text = "Calculation: 2 * 3 * 4 = 24"
        formatted = format_markdown_text(text)
        # 2 * 3 * 4 has spaces after asterisks, so should not turn into <em>
        assert "<em>" not in formatted

    def test_non_string_input(self):
        assert format_markdown_text(12345) == "12345"
        assert format_markdown_text(None) == "None"

