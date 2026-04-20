from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_shell_exposes_neural_pause_and_text_size_accessibility_controls():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert 'id="neural-pause-toggle"' in html
    assert "Pause Neural Stream Visuals" in html
    assert 'id="setting-neural-paused"' in html
    assert 'id="setting-chat-text-size"' in html
    assert 'id="setting-neural-text-size"' in html

    assert "neuralFeedPaused: false" in js
    assert "toggleNeuralVisualPause" in js
    assert "state.neuralFeedMode === 'paused'" in js
    assert "neuralPaused: false" in js
    assert "chatTextSize: 'standard'" in js
    assert "neuralTextSize: 'standard'" in js
    assert "document.body.classList.add(`chat-text-${s.chatTextSize || 'standard'}`)" in js
    assert "document.body.classList.add(`neural-text-${s.neuralTextSize || 'standard'}`)" in js
    assert "document.body.classList.toggle('neural-visual-paused', !!s.neuralPaused)" in js

    assert ".neural-toggle-btn.neural-pause-btn" in css
    assert ".neural-mode-state.paused" in css
    assert "body.chat-text-large" in css
    assert "body.neural-text-large" in css
    assert "body.neural-visual-paused #pane-neural #neural-feed" in css
