"""
Apple Design System & Studio UI for PersonaPlex Voice Orchestration.
Includes:
- Siri Apple Intelligence fluid multi-color glowing orb and dynamic waveform
- RAG document drag-and-drop upload (PDF, TXT, MD, CSV)
- Interactive voice preset gallery with 18 PersonaPlex voices categorized
- Apple iMessage-styled full duplex conversational transcript
- Floating Apple glassmorphism call control bar
- SOLID modular frontend JavaScript architecture
"""

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>PersonaPlex Voice Studio • Apple Design</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #000000;
      --bg-surface: rgba(22, 22, 26, 0.72);
      --bg-elevated: rgba(32, 32, 38, 0.85);
      --bg-pill: rgba(255, 255, 255, 0.08);
      --bg-pill-hover: rgba(255, 255, 255, 0.14);
      --bg-pill-active: rgba(255, 255, 255, 0.22);
      
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-medium: rgba(255, 255, 255, 0.16);
      --border-highlight: rgba(255, 255, 255, 0.28);
      
      --apple-blue: #0071e3;
      --apple-cyan: #2997ff;
      --apple-green: #30d158;
      --apple-purple: #bf5af2;
      --apple-indigo: #5e5ce6;
      --apple-red: #ff453a;
      --apple-orange: #ff9f0a;
      --apple-yellow: #ffd60a;
      
      --text-primary: #f5f5f7;
      --text-secondary: #86868b;
      --text-tertiary: #6e6e73;
      
      --blur-glass: blur(28px) saturate(190%);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 22px;
      --radius-full: 9999px;
      
      --shadow-subtle: 0 4px 20px rgba(0, 0, 0, 0.35);
      --shadow-glow: 0 0 35px rgba(41, 151, 255, 0.25);
      --font-system: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
      --font-mono: "JetBrains Mono", SFMono-Regular, Menlo, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }

    body {
      background-color: var(--bg-base);
      background-image: 
        radial-gradient(circle at 50% 0%, rgba(94, 92, 230, 0.15) 0%, transparent 60%),
        radial-gradient(circle at 10% 20%, rgba(41, 151, 255, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(191, 90, 242, 0.1) 0%, transparent 50%);
      color: var(--text-primary);
      font-family: var(--font-system);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Apple Navigation Bar */
    header {
      height: 54px;
      border-bottom: 0.5px solid var(--border-subtle);
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: var(--blur-glass);
      -webkit-backdrop-filter: var(--blur-glass);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-icon {
      width: 22px;
      height: 22px;
      background: linear-gradient(135deg, var(--apple-cyan), var(--apple-purple));
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 8px rgba(41, 151, 255, 0.4);
    }
    .brand-icon svg { width: 14px; height: 14px; fill: white; }

    .brand-title {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text-primary);
    }

    .brand-tag {
      font-size: 11px;
      font-weight: 500;
      color: var(--text-secondary);
      background: var(--bg-pill);
      padding: 2px 8px;
      border-radius: var(--radius-full);
      border: 0.5px solid var(--border-subtle);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 500;
      padding: 4px 12px;
      border-radius: var(--radius-full);
      background: rgba(255, 255, 255, 0.05);
      border: 0.5px solid var(--border-subtle);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--apple-yellow);
      box-shadow: 0 0 8px currentColor;
    }
    .status-pill.connected .status-dot { background: var(--apple-green); }
    .status-pill.speaking .status-dot { background: var(--apple-cyan); }
    .status-pill.barge-in .status-dot { background: var(--apple-red); }

    /* Main Studio Container */
    .studio-container {
      flex: 1;
      display: grid;
      grid-template-columns: 280px minmax(320px, 1fr) 280px;
      gap: 14px;
      padding: 14px 18px 18px 18px;
      overflow: hidden;
      min-width: 0;
    }

    /* Glass Panels */
    .panel {
      background: var(--bg-surface);
      backdrop-filter: var(--blur-glass);
      -webkit-backdrop-filter: var(--blur-glass);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      box-shadow: var(--shadow-subtle);
      overflow: hidden;
      position: relative;
      min-width: 0;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .panel-title {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Apple Segmented Controls */
    .segmented-control {
      background: rgba(0, 0, 0, 0.45);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 3px;
      display: flex;
      gap: 2px;
      user-select: none;
    }

    .segmented-btn {
      flex: 1;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      background: transparent;
      border: none;
      padding: 6px 8px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      text-align: center;
    }
    .segmented-btn:hover { color: var(--text-primary); }
    .segmented-btn.active {
      background: rgba(255, 255, 255, 0.16);
      color: #ffffff;
      font-weight: 600;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }

    /* Persona List */
    .persona-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      overflow-y: auto;
      max-height: 220px;
      padding-right: 4px;
    }
    .persona-card {
      background: rgba(255, 255, 255, 0.03);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .persona-card:hover {
      background: rgba(255, 255, 255, 0.07);
      border-color: var(--border-medium);
      transform: translateY(-1px);
    }
    .persona-card.active {
      background: rgba(41, 151, 255, 0.12);
      border-color: var(--apple-cyan);
    }
    .persona-avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.02));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      border: 0.5px solid var(--border-subtle);
    }
    .persona-info { flex: 1; min-width: 0; }
    .persona-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
    .persona-role { font-size: 11px; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* System Prompt & Call Rules Card */
    .system-prompt-card {
      background: rgba(0, 0, 0, 0.45);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .prompt-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .prompt-tag {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--apple-cyan);
      font-weight: 500;
    }
    .prompt-badge {
      font-size: 9.5px;
      padding: 2px 7px;
      border-radius: var(--radius-full);
      background: rgba(48, 209, 88, 0.15);
      color: var(--apple-green);
      font-weight: 600;
    }
    .prompt-body {
      font-size: 10.5px;
      line-height: 1.5;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      white-space: pre-wrap;
      max-height: 135px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .doc-badge {
      font-size: 10px;
      color: var(--apple-cyan);
      background: rgba(41, 151, 255, 0.12);
      padding: 2px 6px;
      border-radius: var(--radius-full);
      white-space: nowrap;
    }

    /* Center Stage: Siri Orb, Waveform & Chat */
    .center-panel {
      display: flex;
      flex-direction: column;
      gap: 12px;
      position: relative;
    }

    .visualizer-stage {
      height: 180px;
      background: radial-gradient(circle at center, rgba(30, 30, 45, 0.6) 0%, rgba(10, 10, 15, 0.8) 100%);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    #orb-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    .visualizer-overlay {
      position: relative;
      z-index: 10;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      pointer-events: none;
    }

    .speaker-label {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: #ffffff;
      text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
    }

    .cadence-badge {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--apple-cyan);
      background: rgba(0, 0, 0, 0.5);
      border: 0.5px solid var(--border-subtle);
      padding: 2px 8px;
      border-radius: var(--radius-full);
      backdrop-filter: blur(8px);
    }

    /* Apple iMessage Transcript Container */
    .chat-transcript {
      flex: 1;
      background: rgba(10, 10, 15, 0.6);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .message-row {
      display: flex;
      flex-direction: column;
      max-width: 90%;
      min-width: 0;
      animation: messageSlideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .message-row.agent { align-self: flex-start; }
    .message-row.user { align-self: flex-end; align-items: flex-end; }

    .message-author {
      font-size: 11px;
      font-weight: 500;
      color: var(--text-tertiary);
      margin-bottom: 3px;
      padding: 0 4px;
    }

    .bubble {
      padding: 10px 14px;
      font-size: 13px;
      line-height: 1.45;
      border-radius: 18px;
      word-break: normal;
      overflow-wrap: break-word;
      min-width: 0;
    }

    .message-row.agent .bubble {
      background: rgba(255, 255, 255, 0.08);
      color: #f5f5f7;
      border: 0.5px solid var(--border-subtle);
      border-bottom-left-radius: 4px;
    }
    .message-row.agent.grounded .bubble {
      border-color: rgba(48, 209, 88, 0.4);
      background: rgba(48, 209, 88, 0.08);
    }

    .message-row.user .bubble {
      background: linear-gradient(135deg, var(--apple-blue), var(--apple-cyan));
      color: #ffffff;
      border-bottom-right-radius: 4px;
      box-shadow: 0 2px 10px rgba(0, 113, 227, 0.3);
    }

    /* Floating Call Control Bar */
    .call-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(18, 18, 24, 0.85);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-full);
      padding: 6px 12px;
      box-shadow: var(--shadow-subtle);
      backdrop-filter: var(--blur-glass);
      -webkit-backdrop-filter: var(--blur-glass);
    }

    .chat-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: #fff;
      font-size: 13px;
      font-family: inherit;
      padding: 6px 10px;
    }
    .chat-input::placeholder { color: var(--text-tertiary); }

    .btn-circle {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: 0.5px solid var(--border-subtle);
      background: var(--bg-pill);
      color: var(--text-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-circle:hover { background: var(--bg-pill-hover); transform: scale(1.05); }
    .btn-circle.active { background: rgba(255, 69, 58, 0.2); color: var(--apple-red); border-color: var(--apple-red); }

    .btn-call {
      height: 38px;
      padding: 0 18px;
      border-radius: var(--radius-full);
      border: none;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      background: var(--apple-green);
      color: #000000;
      box-shadow: 0 2px 12px rgba(48, 209, 88, 0.35);
    }
    .btn-call:hover { transform: scale(1.03); opacity: 0.95; }
    .btn-call.end {
      background: var(--apple-red);
      color: #ffffff;
      box-shadow: 0 2px 12px rgba(255, 69, 58, 0.35);
    }

    /* Right Panel: Voice Selection & Observability */
    .voice-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      overflow-y: auto;
      max-height: 250px;
      padding-right: 4px;
    }

    .voice-card {
      background: rgba(255, 255, 255, 0.03);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 8px 10px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 2px;
      transition: all 0.2s ease;
    }
    .voice-card:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: var(--border-medium);
    }
    .voice-card.selected {
      border-color: var(--apple-cyan);
      background: rgba(41, 151, 255, 0.12);
    }
    .voice-id { font-size: 12px; font-weight: 600; color: var(--text-primary); }
    .voice-tag { font-size: 10px; color: var(--apple-cyan); }
    .voice-desc { font-size: 9.5px; color: var(--text-secondary); line-height: 1.2; }

    /* Observability Rows */
    .telemetry-group {
      background: rgba(0, 0, 0, 0.3);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }
    .telemetry-label { color: var(--text-secondary); }
    .telemetry-val { font-family: var(--font-mono); font-weight: 500; color: var(--apple-cyan); }

    /* Keyframes */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes messageSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }
  </style>
</head>
<body>
  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14.5v-9l6 4.5-6 4.5z"/></svg>
      </div>
      <span class="brand-title">PersonaPlex Studio</span>
      <span class="brand-tag">Full-Duplex S2S</span>
    </div>

    <div class="header-actions">
      <div id="status-pill" class="status-pill">
        <span class="status-dot"></span>
        <span id="status-text">Standby</span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <div class="studio-container">
    <!-- Left Column: Personas & RAG Document Knowledge -->
    <div class="panel" style="overflow-y: auto;">
      <div class="panel-header">
        <span class="panel-title">Voice Accent</span>
        <span id="active-accent-badge" class="doc-badge" style="color: var(--apple-cyan);">Indian</span>
      </div>
      <div class="segmented-control" id="accent-control">
        <button class="segmented-btn active" id="accent-btn-indian" onclick="window.studioApp.selectAccent('indian')">Indian</button>
        <button class="segmented-btn" id="accent-btn-american" onclick="window.studioApp.selectAccent('american')">American</button>
        <button class="segmented-btn" id="accent-btn-british" onclick="window.studioApp.selectAccent('british')">British</button>
      </div>

      <div class="panel-header" style="margin-top: 2px;">
        <span class="panel-title">Character</span>
        <span id="active-char-badge" class="doc-badge" style="color: var(--apple-purple);">Professional</span>
      </div>
      <div class="segmented-control" id="character-control">
        <button class="segmented-btn active" id="char-btn-professional" onclick="window.studioApp.selectCharacter('professional')">Professional</button>
        <button class="segmented-btn" id="char-btn-funny" onclick="window.studioApp.selectCharacter('funny')">Funny</button>
        <button class="segmented-btn" id="char-btn-warm" onclick="window.studioApp.selectCharacter('warm')">Warm</button>
      </div>

      <div class="panel-header" style="margin-top: 2px;">
        <span class="panel-title">Active Persona</span>
        <span id="active-preset-badge" class="doc-badge" style="color: var(--apple-green);">Aarav</span>
      </div>

      <div class="persona-list" id="persona-list" style="max-height: 120px;">
        <!-- Dynamically loaded -->
      </div>

      <div class="panel-header" style="margin-top: 6px;">
        <span class="panel-title">System Prompt &amp; Rules</span>
        <span id="principles-badge" class="doc-badge" style="color: var(--apple-cyan);">12 Principles</span>
      </div>

      <div class="system-prompt-card">
        <div class="prompt-header">
          <span class="prompt-tag">Voice Conditioning</span>
          <span class="prompt-badge">Strict Principles</span>
        </div>
        <div class="prompt-body" id="system-prompt-display">1. Listen First
2. Respond to Latest Message
3. Be Concise (1–2 sentences)
4. One Question at a Time
5. Do Not Repeat Known Info
6. Natural Acknowledgements
7. Human-like Turn Taking
8. Never Sound Robotic
9. Handle Interruptions
10. Handle Uncertainty
11. Maintain Context &amp; Goals
12. Priority (A → B → C → D → E)</div>
      </div>

      <div class="panel-header" style="margin-top: 6px;">
        <span class="panel-title">Live Call Context</span>
        <span id="call-flow-state" class="doc-badge" style="color: var(--apple-green);">Standby</span>
      </div>

      <div class="telemetry-group" style="padding: 8px 10px; font-size: 11.5px;">
        <div class="telemetry-row">
          <span class="telemetry-label">Caller Name</span>
          <span class="telemetry-val" id="ctx-caller-name">—</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Detected Goal</span>
          <span class="telemetry-val" id="ctx-caller-goal">Conversational</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Speech Turn Pacing</span>
          <span class="telemetry-val" style="color: var(--apple-cyan);">Concise (1–2 sentences)</span>
        </div>
      </div>
    </div>

    <!-- Center Column: Siri Orb Visualizer & Full-Duplex Transcript -->
    <div class="panel center-panel">
      <!-- Apple Intelligence Siri Orb Visualizer Stage -->
      <div class="visualizer-stage">
        <canvas id="orb-canvas"></canvas>
        <div class="visualizer-overlay">
          <span class="speaker-label" id="speaker-status">Ready to Speak</span>
          <span class="cadence-badge" id="cadence-indicator">24 kHz • 12.5 Hz • Full-Duplex</span>
          <div style="width: 140px; height: 5px; background: rgba(255,255,255,0.12); border-radius: 999px; overflow: hidden; margin-top: 4px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);">
            <div id="mic-meter" style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--apple-green), var(--apple-cyan)); transition: width 0.06s ease;"></div>
          </div>
        </div>
      </div>

      <!-- Transcript Container -->
      <div class="chat-transcript" id="chat-transcript">
        <div class="message-row agent">
          <span class="message-author" id="initial-agent-author">Aarav</span>
          <div class="bubble">Welcome! Choose your preferred accent (Indian, American, British) and character (Professional, Funny, Warm &amp; Concise). The agent follows strict conversation principles—concise turns, active listening, and clean talking.</div>
        </div>
      </div>

      <!-- Bottom Call Controls -->
      <div class="call-controls">
        <button id="btn-mic-toggle" class="btn-circle" title="Toggle Mute" onclick="window.studioApp.toggleMute()">
          <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
          </svg>
        </button>

        <input type="text" id="user-text-input" class="chat-input" placeholder="Speak into mic or type a prompt..." onkeydown="if(event.key==='Enter') window.studioApp.sendTextMessage()">

        <button class="btn-circle" title="Send Text" onclick="window.studioApp.sendTextMessage()">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>

        <button id="btn-call-action" class="btn-call" onclick="window.studioApp.toggleCall()">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M6.62 10.79a15.053 15.053 0 006.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/></svg>
          <span id="call-btn-text">Start Call</span>
        </button>
      </div>
    </div>

    <!-- Right Column: Voice Selection & Telemetry -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">PersonaPlex Voices</span>
      </div>

      <!-- Segmented filter for voices -->
      <div class="segmented-control" id="voice-filter">
        <button class="segmented-btn active" onclick="window.studioApp.filterVoices('All')">All</button>
        <button class="segmented-btn" onclick="window.studioApp.filterVoices('Female')">Female</button>
        <button class="segmented-btn" onclick="window.studioApp.filterVoices('Male')">Male</button>
        <button class="segmented-btn" onclick="window.studioApp.filterVoices('Variety')">Variety</button>
      </div>

      <!-- Voice cards -->
      <div class="voice-grid" id="voice-grid">
        <!-- Dynamically rendered voice cards -->
      </div>

      <div class="panel-header" style="margin-top: 10px;">
        <span class="panel-title">Session Telemetry</span>
      </div>

      <div class="telemetry-group">
        <div class="telemetry-row">
          <span class="telemetry-label">Active Worker</span>
          <span class="telemetry-val" id="telemetry-worker">worker-1</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Mic Inbound Frames</span>
          <span class="telemetry-val" id="telemetry-frames-in">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Agent Outbound Frames</span>
          <span class="telemetry-val" id="telemetry-frames-out">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Barge-in Interruptions</span>
          <span class="telemetry-val" id="telemetry-barge-in">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Noise Canceller</span>
          <span class="telemetry-val" style="color: var(--apple-green);">Active (80Hz + Sub)</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Turn-taking Hangtime</span>
          <span class="telemetry-val">600 ms</span>
        </div>
      </div>
    </div>
  </div>

  <!-- SOLID Modular Architecture JavaScript -->
  <script>
    /**
     * Single Responsibility: Audio Capture, Resampling & Web Audio Playback
     */
    class AudioPipeline {
      constructor(sampleRate = 24000, frameSize = 1920) {
        this.sampleRate = sampleRate;
        this.frameSize = frameSize;
        this.audioCtx = null;
        this.micStream = null;
        this.micSource = null;
        this.processor = null;
        this.silentGain = null;
        this.nextPlayTime = 0;
        this.micBuffer = [];
        this.isMuted = false;
        this.onFrameCallback = null;
        this.onEnergyCallback = null;
      }

      async initialize() {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        try {
          this.audioCtx = new AudioContextClass({ sampleRate: this.sampleRate });
          if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
          }
        } catch (e) {
          this.audioCtx = new AudioContextClass();
        }
        this.nextPlayTime = this.audioCtx.currentTime;
      }

      async startMicrophone(onFrame, onEnergy) {
        this.onFrameCallback = onFrame;
        this.onEnergyCallback = onEnergy;
        await this.initialize();

        if (this.audioCtx && this.audioCtx.state === 'suspended') {
          await this.audioCtx.resume();
        }

        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });

        this.micSource = this.audioCtx.createMediaStreamSource(this.micStream);
        this.processor = this.audioCtx.createScriptProcessor(2048, 1, 1);
        window._activeAudioProcessor = this.processor; // prevent Chromium GC
        this.silentGain = this.audioCtx.createGain();
        this.silentGain.gain.value = 0.0;

        const inRate = this.audioCtx.sampleRate;

        this.processor.onaudioprocess = (e) => {
          if (this.isMuted) return;
          const input = e.inputBuffer.getChannelData(0);

          // Compute instantaneous RMS
          let sumSquares = 0;
          for (let i = 0; i < input.length; i++) {
            sumSquares += input[i] * input[i];
          }
          const rms = Math.sqrt(sumSquares / input.length);
          if (this.onEnergyCallback) this.onEnergyCallback(rms);

          // Resample to target 24kHz
          let resampled;
          if (inRate === this.sampleRate) {
            resampled = input;
          } else {
            const ratio = this.sampleRate / inRate;
            const newLen = Math.round(input.length * ratio);
            resampled = new Float32Array(newLen);
            for (let i = 0; i < newLen; i++) {
              const src = i / ratio;
              const idx0 = Math.floor(src);
              const idx1 = Math.min(idx0 + 1, input.length - 1);
              const frac = src - idx0;
              resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
            }
          }

          for (let i = 0; i < resampled.length; i++) {
            this.micBuffer.push(resampled[i]);
          }

          while (this.micBuffer.length >= this.frameSize) {
            const frame = new Float32Array(this.micBuffer.slice(0, this.frameSize));
            this.micBuffer = this.micBuffer.slice(this.frameSize);
            if (this.onFrameCallback) this.onFrameCallback(frame);
          }
        };

        this.micSource.connect(this.processor);
        this.processor.connect(this.silentGain);
        this.silentGain.connect(this.audioCtx.destination);
      }

      playChunk(floatSamples) {
        if (!this.audioCtx) return;
        const buffer = this.audioCtx.createBuffer(1, floatSamples.length, this.sampleRate);
        buffer.copyToChannel(floatSamples, 0);

        const source = this.audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioCtx.destination);

        const now = this.audioCtx.currentTime;
        if (this.nextPlayTime < now) {
          this.nextPlayTime = now + 0.02;
        }

        source.start(this.nextPlayTime);
        this.nextPlayTime += buffer.duration;
      }

      stopPlayback() {
        if (this.audioCtx) {
          this.nextPlayTime = this.audioCtx.currentTime;
        }
      }

      toggleMute() {
        this.isMuted = !this.isMuted;
        return this.isMuted;
      }

      stop() {
        if (this.micStream) {
          this.micStream.getTracks().forEach(t => t.stop());
          this.micStream = null;
        }
        if (this.processor) {
          try { this.processor.disconnect(); } catch (e) {}
          this.processor = null;
        }
        if (this.micSource) {
          try { this.micSource.disconnect(); } catch (e) {}
          this.micSource = null;
        }
        this.micBuffer = [];
      }
    }

    /**
     * Single Responsibility: Browser Speech-to-Text Recognition
     */
    class SpeechRecognizer {
      constructor(onTranscript, onError) {
        this.onTranscript = onTranscript;
        this.onError = onError;
        this.recognition = null;
        this.isRunning = false;
        this.init();
      }

      init() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          console.warn('SpeechRecognition API not available in this browser');
          return;
        }

        try {
          this.recognition = new SpeechRecognition();
          this.recognition.continuous = true;
          this.recognition.interimResults = true;
          this.recognition.lang = 'en-US';

          this.recognition.onresult = (event) => {
            let interim = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
              const res = event.results[i];
              const text = res[0].transcript;
              if (res.isFinal) {
                if (text.trim()) {
                  this.onTranscript(text.trim(), true);
                }
              } else {
                interim += text;
              }
            }
            if (interim.trim()) {
              this.onTranscript(interim.trim(), false);
            }
          };

          this.recognition.onerror = (e) => {
            console.warn('Speech recognition status:', e.error);
            if (this.onError) this.onError(e);
          };

          this.recognition.onend = () => {
            if (this.isRunning) {
              try { this.recognition.start(); } catch (e) {}
            }
          };
        } catch (e) {
          console.warn('Speech recognition init error:', e);
        }
      }

      start() {
        if (!this.recognition) return;
        this.isRunning = true;
        try {
          this.recognition.start();
        } catch (e) {}
      }

      setLang(lang) {
        if (this.recognition) {
          this.recognition.lang = lang;
        }
      }

      stop() {
        this.isRunning = false;
        if (this.recognition) {
          try {
            this.recognition.stop();
          } catch (e) {}
        }
      }
    }

    /**
     * Single Responsibility: Full-Duplex Binary WebSocket Protocol Client
     */
    class VoiceSocket {
      constructor() {
        this.ws = null;
        this.onAudio = null;
        this.onText = null;
        this.onMetadata = null;
        this.onError = null;
        this.onOpen = null;
        this.onClose = null;
      }

      connect(url) {
        this.ws = new WebSocket(url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          if (this.onOpen) this.onOpen();
        };

        this.ws.onmessage = (event) => {
          const data = new Uint8Array(event.data);
          if (data.length === 0) return;
          const kind = data[0];
          const payload = data.slice(1);

          if (kind === 0x01 && this.onAudio) {
            const floatSamples = new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4);
            this.onAudio(floatSamples);
          } else if (kind === 0x02 && this.onText) {
            const text = new TextDecoder().decode(payload);
            this.onText(text);
          } else if (kind === 0x04 && this.onMetadata) {
            const meta = JSON.parse(new TextDecoder().decode(payload));
            this.onMetadata(meta);
          } else if (kind === 0x05 && this.onError) {
            const err = new TextDecoder().decode(payload);
            this.onError(err);
          }
        };

        this.ws.onclose = () => {
          if (this.onClose) this.onClose();
        };
      }

      sendAudioFrame(float32Array) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const msg = new Uint8Array(1 + float32Array.byteLength);
        msg[0] = 0x01; // Audio frame
        msg.set(new Uint8Array(float32Array.buffer), 1);
        this.ws.send(msg);
      }

      sendTextMessage(text) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const textBytes = new TextEncoder().encode(text);
        const msg = new Uint8Array(1 + textBytes.byteLength);
        msg[0] = 0x02; // Text token
        msg.set(textBytes, 1);
        this.ws.send(msg);
      }

      sendControl(actionByte = 0x01) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const msg = new Uint8Array([0x03, actionByte]);
        this.ws.send(msg);
      }

      disconnect() {
        if (this.ws) {
          this.ws.close();
          this.ws = null;
        }
      }
    }

    /**
     * Single Responsibility: System Prompt & Call Flow Principles
     */
    class SystemPromptManager {
      async getSystemPrompt() {
        try {
          const res = await fetch('/v1/system-prompt');
          return await res.json();
        } catch (e) {
          return { prompt: "12 Conversation Principles Active" };
        }
      }
    }

    /**
     * Single Responsibility: Siri Apple Intelligence Dynamic Orb & Waveform Renderer
     */
    class OrbRenderer {
      constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width = canvas.offsetWidth;
        this.height = canvas.height = canvas.offsetHeight;
        this.phase = 0;
        this.energy = 0;
        this.agentSpeaking = false;
        this.animationId = null;

        window.addEventListener('resize', () => {
          this.width = this.canvas.width = this.canvas.offsetWidth;
          this.height = this.canvas.height = this.canvas.offsetHeight;
        });
      }

      start() {
        const render = () => {
          this.draw();
          this.animationId = requestAnimationFrame(render);
        };
        this.animationId = requestAnimationFrame(render);
      }

      setEnergy(e) {
        this.energy = Math.max(0.05, Math.min(1.0, e * 6));
      }

      setAgentSpeaking(isSpeaking) {
        this.agentSpeaking = isSpeaking;
      }

      draw() {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;
        ctx.clearRect(0, 0, w, h);

        this.phase += this.agentSpeaking ? 0.08 : 0.03;
        const cx = w / 2;
        const cy = h / 2;
        const baseRadius = 38 + (this.energy * 24) + (this.agentSpeaking ? Math.sin(this.phase * 2) * 8 : 0);

        // Multi-layered Siri glow
        const gradient = ctx.createRadialGradient(cx, cy, baseRadius * 0.2, cx, cy, baseRadius * 2.2);
        if (this.agentSpeaking) {
          gradient.addColorStop(0, 'rgba(48, 209, 88, 0.85)'); // Apple Green
          gradient.addColorStop(0.4, 'rgba(41, 151, 255, 0.6)'); // Apple Cyan
          gradient.addColorStop(0.7, 'rgba(191, 90, 242, 0.4)'); // Purple
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        } else if (this.energy > 0.12) {
          gradient.addColorStop(0, 'rgba(41, 151, 255, 0.85)');
          gradient.addColorStop(0.5, 'rgba(94, 92, 230, 0.5)');
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        } else {
          gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
          gradient.addColorStop(0.6, 'rgba(41, 151, 255, 0.15)');
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(cx, cy, baseRadius * 2.2, 0, Math.PI * 2);
        ctx.fill();

        // Harmonic Waveform lines
        ctx.save();
        ctx.lineWidth = 2.2;
        const waves = 3;
        for (let j = 0; j < waves; j++) {
          ctx.beginPath();
          const wavePhase = this.phase + (j * Math.PI) / 2;
          const strokeAlpha = 0.3 + (j * 0.25);
          ctx.strokeStyle = this.agentSpeaking
            ? `rgba(48, 209, 88, ${strokeAlpha})`
            : `rgba(41, 151, 255, ${strokeAlpha})`;

          for (let x = 0; x < w; x += 4) {
            const relX = (x - cx) / (w * 0.4);
            const envelope = Math.exp(-relX * relX * 2);
            const amp = (this.energy * 28 + (this.agentSpeaking ? 22 : 8)) * envelope;
            const y = cy + Math.sin(x * 0.02 + wavePhase) * amp;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    /**
     * Single Responsibility: Main Controller coordinating UI, RAG, Audio & Persona
     */
    class StudioApp {
      constructor() {
        this.audio = new AudioPipeline();
        this.socket = new VoiceSocket();
        this.orb = null;

        this.activeAccent = 'indian';
        this.activeCharacter = 'professional';
        this.activePersona = 'indian_pro';
        this.activeAgentName = 'Aarav';
        this.activeVoice = 'NATM0.pt';
        this.isConnected = false;
        this.agentSpeaking = false;
        this.currentVoiceFilter = 'All';

        this.framesIn = 0;
        this.framesOut = 0;
        this.bargeIns = 0;

        this.personas = [];
        this.voices = [];
        this.promptManager = new SystemPromptManager();

        this.recognizer = new SpeechRecognizer(
          (text, isFinal) => this.handleSpeechRecognized(text, isFinal),
          (err) => console.log('Recognizer info:', err)
        );

        this.initElements();
        this.initSocketEvents();
      }

      initElements() {
        const canvas = document.getElementById('orb-canvas');
        this.orb = new OrbRenderer(canvas);
        this.orb.start();
      }

      initSocketEvents() {
        this.socket.onOpen = () => {
          this.isConnected = true;
          this.updateStatus('Active • Full-Duplex', 'connected');
          const callBtn = document.getElementById('btn-call-action');
          callBtn.classList.add('end');
          document.getElementById('call-btn-text').innerText = 'End Call';
          document.getElementById('speaker-status').innerText = `Listening (${this.activeAgentName} ready)`;
          const stateBadge = document.getElementById('call-flow-state');
          if (stateBadge) stateBadge.innerText = 'Active Call';
        };

        this.socket.onAudio = (floatSamples) => {
          this.framesOut++;
          document.getElementById('telemetry-frames-out').innerText = this.framesOut;
          this.agentSpeaking = true;
          this.orb.setAgentSpeaking(true);
          document.getElementById('speaker-status').innerText = `${this.activeAgentName} Speaking`;
          this.audio.playChunk(floatSamples);
        };

        this.socket.onText = (token) => {
          this.appendTranscriptToken('agent', token);
        };

        this.socket.onMetadata = (meta) => {
          if (meta.event === 'barge_in') {
            this.bargeIns++;
            document.getElementById('telemetry-barge-in').innerText = this.bargeIns;
            this.audio.stopPlayback();
            this.agentSpeaking = false;
            this.orb.setAgentSpeaking(false);
            document.getElementById('speaker-status').innerText = 'Interrupted (Barge-in)';
          }
        };

        this.socket.onError = (err) => {
          alert('Session Error: ' + err);
          this.disconnect();
        };

        this.socket.onClose = () => {
          this.disconnect();
        };
      }

      async boot() {
        await this.loadPersonas();
        await this.loadVoices();
        await this.loadSystemPrompt();
        this.syncMatrixSelection();
      }

      async loadSystemPrompt() {
        const data = await this.promptManager.getSystemPrompt();
        const display = document.getElementById('system-prompt-display');
        if (display && data.prompt) {
          display.innerText = data.prompt;
        }
      }

      selectAccent(accent) {
        this.activeAccent = accent;
        this.updateAccentButtons();
        this.syncMatrixSelection();
      }

      selectCharacter(character) {
        this.activeCharacter = character;
        this.updateCharacterButtons();
        this.syncMatrixSelection();
      }

      updateAccentButtons() {
        ['indian', 'american', 'british'].forEach(acc => {
          const btn = document.getElementById(`accent-btn-${acc}`);
          if (btn) btn.classList.toggle('active', acc === this.activeAccent);
        });
        const badge = document.getElementById('active-accent-badge');
        if (badge) {
          badge.innerText = this.activeAccent.charAt(0).toUpperCase() + this.activeAccent.slice(1);
        }
        const langMap = { indian: 'en-IN', american: 'en-US', british: 'en-GB' };
        if (this.recognizer) {
          this.recognizer.setLang(langMap[this.activeAccent] || 'en-US');
        }
      }

      updateCharacterButtons() {
        ['professional', 'funny', 'warm'].forEach(ch => {
          const btn = document.getElementById(`char-btn-${ch}`);
          if (btn) btn.classList.toggle('active', ch === this.activeCharacter);
        });
        const badge = document.getElementById('active-char-badge');
        if (badge) {
          badge.innerText = this.activeCharacter === 'warm' ? 'Warm & Concise' : (this.activeCharacter.charAt(0).toUpperCase() + this.activeCharacter.slice(1));
        }
      }

      syncMatrixSelection() {
        const matrix = {
          indian: {
            professional: { id: 'indian_pro', name: 'Aarav', voice: 'NATM0.pt' },
            funny: { id: 'indian_funny', name: 'Rohan', voice: 'NATM1.pt' },
            warm: { id: 'indian_warm', name: 'Ananya', voice: 'NATF0.pt' }
          },
          american: {
            professional: { id: 'american_pro', name: 'Sarah', voice: 'NATF1.pt' },
            funny: { id: 'american_funny', name: 'Jack', voice: 'NATM2.pt' },
            warm: { id: 'american_warm', name: 'Maya', voice: 'NATF2.pt' }
          },
          british: {
            professional: { id: 'british_pro', name: 'Arthur', voice: 'NATM3.pt' },
            funny: { id: 'british_funny', name: 'Oliver', voice: 'NATM0.pt' },
            warm: { id: 'british_warm', name: 'Emma', voice: 'NATF3.pt' }
          }
        };

        const target = matrix[this.activeAccent]?.[this.activeCharacter] || matrix.indian.professional;
        this.activePersona = target.id;
        this.activeAgentName = target.name;
        this.activeVoice = target.voice;

        const presetBadge = document.getElementById('active-preset-badge');
        if (presetBadge) presetBadge.innerText = target.name;

        const initialAuthor = document.getElementById('initial-agent-author');
        if (initialAuthor && !this.isConnected) initialAuthor.innerText = target.name;

        this.renderPersonas();
        this.renderVoices();
      }

      async loadPersonas() {
        try {
          const res = await fetch('/v1/agents');
          const data = await res.json();
          this.personas = data.agents || [];
          this.renderPersonas();
        } catch (e) {
          console.error('Failed loading personas', e);
        }
      }

      renderPersonas() {
        const container = document.getElementById('persona-list');
        if (!container) return;
        container.innerHTML = this.personas.map(p => `
          <div class="persona-card ${p.id === this.activePersona ? 'active' : ''}" onclick="window.studioApp.selectPersona('${p.id}')">
            <div class="persona-avatar">✦</div>
            <div class="persona-info">
              <div class="persona-name">${p.name}</div>
              <div class="persona-role">${p.description}</div>
            </div>
          </div>
        `).join('');
      }

      selectPersona(personaId) {
        this.activePersona = personaId;
        const p = this.personas.find(item => item.id === personaId);
        if (p) {
          if (p.accent) this.activeAccent = p.accent;
          if (p.character) this.activeCharacter = p.character;
          this.activeVoice = p.voice_prompt;
          this.activeAgentName = p.name.split(' ')[0];
          this.updateAccentButtons();
          this.updateCharacterButtons();
          const presetBadge = document.getElementById('active-preset-badge');
          if (presetBadge) presetBadge.innerText = this.activeAgentName;
          this.renderPersonas();
          this.renderVoices();
        }
      }

      async loadVoices() {
        try {
          const res = await fetch('/v1/voices');
          const data = await res.json();
          this.voices = data.voices || [];
          this.renderVoices();
        } catch (e) {
          console.error('Failed loading voices', e);
        }
      }

      filterVoices(category) {
        this.currentVoiceFilter = category;
        const btns = document.querySelectorAll('#voice-filter .segmented-btn');
        btns.forEach(b => b.classList.toggle('active', b.innerText === category));
        this.renderVoices();
      }

      renderVoices() {
        const container = document.getElementById('voice-grid');
        const filtered = this.voices.filter(v => {
          if (this.currentVoiceFilter === 'All') return true;
          if (this.currentVoiceFilter === 'Female') return v.gender === 'Female';
          if (this.currentVoiceFilter === 'Male') return v.gender === 'Male';
          if (this.currentVoiceFilter === 'Variety') return v.category === 'Variety';
          return true;
        });

        container.innerHTML = filtered.map(v => `
          <div class="voice-card ${v.id === this.activeVoice ? 'selected' : ''}" onclick="window.studioApp.selectVoice('${v.id}')">
            <div class="voice-id">${v.id.replace('.pt', '')}</div>
            <div class="voice-tag">${v.tag}</div>
            <div class="voice-desc">${v.description}</div>
          </div>
        `).join('');
      }

      selectVoice(voiceId) {
        this.activeVoice = voiceId;
        this.renderVoices();
      }

      updateStatus(text, className) {
        const pill = document.getElementById('status-pill');
        pill.className = `status-pill ${className || ''}`;
        document.getElementById('status-text').innerText = text;
      }

      async toggleCall() {
        if (this.isConnected) {
          this.disconnect();
        } else {
          await this.startCall();
        }
      }

      handleSpeechRecognized(text, isFinal) {
        if (!this.isConnected) return;
        this.updateUserLiveBubble(text, isFinal);

        // Live Context Tracking (Principles 5 & 11)
        const nameMatch = text.match(/\b(?:my name is|i am|this is|call me)\s+([A-Z][a-z]+|[a-z]+)\b/i);
        if (nameMatch) {
          const nm = nameMatch[1].charAt(0).toUpperCase() + nameMatch[1].slice(1);
          const el = document.getElementById('ctx-caller-name');
          if (el && !['Here', 'Good', 'Fine', 'Okay', 'Ready'].includes(nm)) el.innerText = nm;
        }
        const goalMatch = text.match(/\b(?:i want to|i need to|looking to|help me with)\s+(.+)/i);
        if (goalMatch) {
          const el = document.getElementById('ctx-caller-goal');
          if (el) el.innerText = goalMatch[1].slice(0, 24);
        }

        if (isFinal) {
          // Immediate barge-in cutoff
          this.audio.stopPlayback();
          this.agentSpeaking = false;
          this.orb.setAgentSpeaking(false);
          document.getElementById('speaker-status').innerText = 'Processing speech...';

          // Send recognized utterance as 0x02 text packet over WebSocket
          this.socket.sendTextMessage(text);
        }
      }

      updateUserLiveBubble(text, isFinal) {
        const transcript = document.getElementById('chat-transcript');
        let liveRow = document.getElementById('live-user-bubble');
        if (!liveRow) {
          liveRow = document.createElement('div');
          liveRow.id = 'live-user-bubble';
          liveRow.className = 'message-row user';
          liveRow.innerHTML = `
            <span class="message-author">You</span>
            <div class="bubble"></div>
          `;
          transcript.appendChild(liveRow);
        }
        const bubble = liveRow.querySelector('.bubble');
        bubble.innerText = text;
        if (isFinal) {
          liveRow.removeAttribute('id'); // Finalize bubble
        }
        transcript.scrollTop = transcript.scrollHeight;
      }

      async startCall() {
        this.updateStatus(`Connecting (${this.activeAgentName})...`, 'connecting');

        try {
          await this.audio.startMicrophone(
            (frame) => {
              this.framesIn++;
              document.getElementById('telemetry-frames-in').innerText = this.framesIn;
              this.socket.sendAudioFrame(frame);
            },
            (energy) => {
              this.orb.setEnergy(energy);
              const meter = document.getElementById('mic-meter');
              if (meter) {
                const pct = Math.min(100, Math.round(energy * 700));
                meter.style.width = `${pct}%`;
              }
            }
          );
        } catch (err) {
          alert('Microphone access denied: ' + err.message);
          this.disconnect();
          return;
        }

        // Start speech recognition with accent language code
        const langMap = { indian: 'en-IN', american: 'en-US', british: 'en-GB' };
        this.recognizer.setLang(langMap[this.activeAccent] || 'en-US');
        this.recognizer.start();

        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${window.location.host}/v1/realtime?persona_id=${encodeURIComponent(this.activePersona)}&voice_prompt=${encodeURIComponent(this.activeVoice)}&accent=${encodeURIComponent(this.activeAccent)}&character=${encodeURIComponent(this.activeCharacter)}`;
        this.socket.connect(url);
      }

      toggleMute() {
        const isMuted = this.audio.toggleMute();
        const btn = document.getElementById('btn-mic-toggle');
        btn.classList.toggle('active', isMuted);
      }

      sendTextMessage() {
        const input = document.getElementById('user-text-input');
        const text = input.value.trim();
        if (!text) return;
        input.value = '';

        this.appendMessage('user', text);
        this.audio.stopPlayback();
        this.orb.setAgentSpeaking(false);
        this.socket.sendTextMessage(text);
      }

      appendMessage(author, text, isGrounded = false) {
        const transcript = document.getElementById('chat-transcript');
        const row = document.createElement('div');
        row.className = `message-row ${author} ${isGrounded ? 'grounded' : ''}`;
        const authorName = author === 'user' ? 'You' : (this.activeAgentName || 'Agent');
        row.innerHTML = `
          <span class="message-author">${authorName}</span>
          <div class="bubble">${text}</div>
        `;
        transcript.appendChild(row);
        transcript.scrollTop = transcript.scrollHeight;
      }

      appendTranscriptToken(author, token) {
        const transcript = document.getElementById('chat-transcript');
        let lastRow = transcript.lastElementChild;
        if (!lastRow || !lastRow.classList.contains(author) || lastRow.id === 'live-user-bubble') {
          lastRow = document.createElement('div');
          lastRow.className = `message-row ${author}`;
          const authorName = author === 'user' ? 'You' : (this.activeAgentName || 'Agent');
          lastRow.innerHTML = `
            <span class="message-author">${authorName}</span>
            <div class="bubble"></div>
          `;
          transcript.appendChild(lastRow);
        }
        const bubble = lastRow.querySelector('.bubble');
        bubble.innerText += token;
        transcript.scrollTop = transcript.scrollHeight;
      }

      disconnect() {
        this.socket.disconnect();
        this.audio.stop();
        this.recognizer.stop();

        this.isConnected = false;
        this.agentSpeaking = false;
        this.orb.setAgentSpeaking(false);
        this.orb.setEnergy(0);

        const meter = document.getElementById('mic-meter');
        if (meter) meter.style.width = '0%';

        this.updateStatus('Standby', '');
        const stateBadge = document.getElementById('call-flow-state');
        if (stateBadge) stateBadge.innerText = 'Standby';
        const callBtn = document.getElementById('btn-call-action');
        callBtn.classList.remove('end');
        document.getElementById('call-btn-text').innerText = 'Start Call';
        document.getElementById('speaker-status').innerText = 'Ready to Speak';
      }
    }

    // Launch Application
    window.addEventListener('DOMContentLoaded', () => {
      window.studioApp = new StudioApp();
      window.studioApp.boot();
    });
  </script>
</body>
</html>
"""
