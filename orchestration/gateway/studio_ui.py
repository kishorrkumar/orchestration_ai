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

    /* Top-Right Header Save Button with Apple Color Changing Transition */
    .btn-header-save {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
      color: #f5f5f7;
      background: rgba(0, 113, 227, 0.22);
      border: 1px solid rgba(41, 151, 255, 0.45);
      border-radius: var(--radius-full);
      cursor: pointer;
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      box-shadow: 0 2px 10px rgba(0, 113, 227, 0.2);
    }
    .btn-header-save:hover {
      background: rgba(0, 113, 227, 0.38);
      border-color: var(--apple-cyan);
      box-shadow: 0 4px 16px rgba(41, 151, 255, 0.35);
      transform: translateY(-1px);
    }
    .btn-header-save:active {
      transform: scale(0.97);
    }
    .btn-header-save svg {
      transition: transform 0.25s ease;
    }
    /* Dynamic Saved State: Color Shifts to Vibrant Apple Emerald Green */
    .btn-header-save.saved {
      background: var(--apple-green) !important;
      border-color: var(--apple-green) !important;
      color: #000000 !important;
      font-weight: 700 !important;
      box-shadow: 0 0 22px rgba(48, 209, 88, 0.7), 0 2px 10px rgba(48, 209, 88, 0.4) !important;
      transform: scale(1.03);
    }
    .btn-header-save.saved svg {
      stroke: #000000 !important;
      stroke-width: 2.5;
      transform: scale(1.1);
    }

    /* Single Voice Profile Card (Locked to One Voice & Tone) */
    .single-agent-profile {
      background: linear-gradient(135deg, rgba(41, 151, 255, 0.12), rgba(94, 92, 230, 0.08));
      border: 1px solid rgba(41, 151, 255, 0.35);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .profile-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .profile-name {
      font-size: 14px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.01em;
    }
    .profile-badge {
      font-size: 10px;
      color: var(--apple-cyan);
      background: rgba(41, 151, 255, 0.15);
      border: 0.5px solid rgba(41, 151, 255, 0.3);
      padding: 2px 7px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-weight: 500;
    }
    .profile-desc {
      font-size: 11.5px;
      color: var(--text-secondary);
      line-height: 1.4;
    }
    .profile-meta-row {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--apple-cyan);
      font-family: var(--font-mono);
      margin-top: 2px;
    }

    /* Customizer Form & Save Agent */
    .customizer-form {
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: rgba(0, 0, 0, 0.25);
      border: 0.5px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 10px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .form-row {
      display: flex;
      gap: 8px;
    }
    .flex-1 { flex: 1; }
    .form-label {
      font-size: 10px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .form-input, .form-select, .form-textarea {
      background: rgba(255, 255, 255, 0.05);
      border: 0.5px solid var(--border-subtle);
      border-radius: 6px;
      color: var(--text-primary);
      font-family: inherit;
      font-size: 12px;
      padding: 6px 8px;
      outline: none;
      transition: all 0.2s ease;
    }
    .form-input:focus, .form-select:focus, .form-textarea:focus {
      border-color: var(--apple-cyan);
      background: rgba(255, 255, 255, 0.08);
      box-shadow: 0 0 8px rgba(41, 151, 255, 0.2);
    }
    .form-select option {
      background: #1c1c1e;
      color: #fff;
    }
    .form-textarea {
      resize: vertical;
      font-family: var(--font-mono);
      font-size: 10.5px;
      line-height: 1.4;
      min-height: 75px;
    }
    .btn-save-agent {
      background: linear-gradient(135deg, var(--apple-cyan), var(--apple-blue));
      color: #ffffff;
      border: none;
      border-radius: 7px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      box-shadow: 0 2px 8px rgba(0, 113, 227, 0.3);
    }
    .btn-save-agent:hover {
      opacity: 0.95;
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(0, 113, 227, 0.45);
    }
    .btn-save-agent:active {
      transform: scale(0.98);
    }
    .save-feedback {
      font-size: 11px;
      font-weight: 600;
      color: var(--apple-green);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    .save-feedback.show {
      opacity: 1;
    }

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
      <button id="btn-header-save" class="btn-header-save" onclick="window.studioApp.saveActiveAgent()" title="Save Agent Profile &amp; Rules">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
        <span id="btn-header-save-text">Save Agent</span>
      </button>

      <div id="status-pill" class="status-pill">
        <span class="status-dot"></span>
        <span id="status-text">Standby</span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <div class="studio-container">
    <!-- Left Column: Indian Voice Agent Customizer (Locked to 1 Voice & 1 Tone) -->
    <div class="panel" style="overflow-y: auto;">
      <div class="panel-header">
        <span class="panel-title">Voice Agent</span>
        <span id="active-agent-badge" class="doc-badge" style="color: var(--apple-cyan);">Aarav</span>
      </div>

      <!-- Locked Single Voice & Tone Profile (Aarav • Indian English) -->
      <div class="single-agent-profile">
        <div class="profile-top">
          <span class="profile-name">Aarav</span>
          <span class="profile-badge">Locked Voice &amp; Tone</span>
        </div>
        <div class="profile-desc">Natural, articulate, and crystal-clear Indian English conversational voice agent.</div>
        <div class="profile-meta-row">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
          <span>en-IN-PrabhatNeural (24 kHz PCM)</span>
        </div>
      </div>

      <!-- Agent Settings & Live Customizer -->
      <div class="panel-header" style="margin-top: 4px;">
        <span class="panel-title">Agent Settings</span>
        <span id="agent-save-feedback" class="save-feedback"></span>
      </div>

      <div class="customizer-form">
        <div class="form-group">
          <label class="form-label" for="cfg-agent-name">Agent Display Name</label>
          <input type="text" id="cfg-agent-name" class="form-input" value="Aarav (Indian English • Articulate &amp; Natural)" placeholder="Agent Name">
        </div>

        <div class="form-row">
          <div class="form-group flex-1">
            <label class="form-label">Voice Tone</label>
            <input type="text" class="form-input" value="Natural &amp; Articulate" readonly style="opacity: 0.85; cursor: default;">
          </div>
          <div class="form-group flex-1">
            <label class="form-label">Neural Voice</label>
            <input type="text" id="cfg-neural-voice" class="form-input" value="en-IN-PrabhatNeural" readonly style="opacity: 0.85; cursor: default; font-family: var(--font-mono); font-size: 11px;">
          </div>
        </div>

        <div class="form-group">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <label class="form-label" for="cfg-system-prompt">System Prompt &amp; Rules</label>
            <span style="font-size:10px; color:var(--apple-green);">12 Principles Enforced</span>
          </div>
          <textarea id="cfg-system-prompt" class="form-textarea" rows="8" placeholder="Agent system prompt..."></textarea>
        </div>
      </div>

      <div class="panel-header" style="margin-top: 4px;">
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
          <div class="bubble">Namaste! I'm Aarav, your Indian English AI voice assistant. I am ready to converse naturally and clearly with you. Click &quot;Start Call&quot; or type below to begin.</div>
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

    <!-- Right Column: Indian Neural Voice Engine & Conversation Rules -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Acoustic Engine</span>
        <span class="doc-badge" style="color: var(--apple-cyan);">Indian Neural</span>
      </div>

      <div class="telemetry-group" style="padding: 10px; font-size: 11.5px;">
        <div class="telemetry-row">
          <span class="telemetry-label">Active Neural Voice</span>
          <span class="telemetry-val" id="active-neural-voice-badge">en-IN-PrabhatNeural</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Accent / Dialect</span>
          <span class="telemetry-val">Indian English (en-IN)</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Sampling Quality</span>
          <span class="telemetry-val">24,000 Hz Mono Float32</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Frame Cadence</span>
          <span class="telemetry-val">12.5 Hz (80ms turns)</span>
        </div>
      </div>

      <div class="panel-header" style="margin-top: 8px;">
        <span class="panel-title">12 Conversation Principles</span>
        <span class="doc-badge" style="color: var(--apple-green);">Active</span>
      </div>

      <div class="system-prompt-card" style="max-height: 180px; overflow-y: auto;">
        <div class="prompt-body" style="font-size: 10.5px; line-height: 1.45;">1. Listen First (Understand before reply)
2. Respond to Latest Message
3. Concise (1–2 sentences)
4. One Question at a Time
5. Do Not Repeat Known Info
6. Natural Acknowledgement
7. Human-like Turn Taking
8. Never Sound Robotic (Zero filler)
9. Handle Interruptions (Instant cut)
10. Handle Uncertainty Gracefully
11. Maintain Context &amp; Goals
12. Truthful &amp; Non-hallucinatory</div>
      </div>

      <div class="panel-header" style="margin-top: 8px;">
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

        // Skip pure silence frames so nextPlayTime doesn't wander into future
        let sumSq = 0;
        for (let i = 0; i < floatSamples.length; i++) {
          sumSq += floatSamples[i] * floatSamples[i];
        }
        const rms = Math.sqrt(sumSq / floatSamples.length);
        if (rms < 0.0008) return;

        const now = this.audioCtx.currentTime;
        if (this.nextPlayTime < now || this.nextPlayTime > now + 0.35) {
          // Snap directly to current time with 15ms buffer for immediate speech
          this.nextPlayTime = now + 0.015;
        }

        const buffer = this.audioCtx.createBuffer(1, floatSamples.length, this.sampleRate);
        buffer.copyToChannel(floatSamples, 0);

        const source = this.audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioCtx.destination);

        source.start(this.nextPlayTime);
        this.nextPlayTime += buffer.duration;
      }

      stopPlayback() {
        if (this.audioCtx) {
          this.nextPlayTime = 0;
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
          this.recognition.lang = 'en-IN';

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
        this.currentAgentBubble = null;

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
          this.currentAgentBubble = null;
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
            this.currentAgentBubble = null;
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
        this.activePersona = 'indian_pro';
        this.activeAgentName = 'Aarav';
        this.activeNeuralVoice = 'en-IN-PrabhatNeural';
        this.activeVoice = 'NATM0.pt';
        this.activeCharacter = 'Professional';
        await this.loadPersonas();
      }

      async loadPersonas() {
        try {
          const res = await fetch('/v1/agents');
          const data = await res.json();
          this.personas = data.agents || [];
          this.populateCustomizer();
        } catch (e) {
          console.error('Failed loading personas', e);
        }
      }

      selectAgentPreset(presetId) {
        this.activePersona = 'indian_pro';
        this.populateCustomizer();
      }

      populateCustomizer() {
        const p = this.personas.find(item => item.id === 'indian_pro') || this.personas[0];
        if (!p) return;

        this.activeAgentName = 'Aarav';
        this.activeCharacter = 'Professional';
        this.activeNeuralVoice = 'en-IN-PrabhatNeural';
        this.activeVoice = 'NATM0.pt';

        const badge = document.getElementById('active-agent-badge');
        if (badge) badge.innerText = 'Aarav';

        const neuralBadge = document.getElementById('active-neural-voice-badge');
        if (neuralBadge) neuralBadge.innerText = 'en-IN-PrabhatNeural';

        const nameInput = document.getElementById('cfg-agent-name');
        if (nameInput) nameInput.value = p.name || 'Aarav (Indian English • Articulate & Natural)';

        const voiceInput = document.getElementById('cfg-neural-voice');
        if (voiceInput) voiceInput.value = 'en-IN-PrabhatNeural';

        const promptArea = document.getElementById('cfg-system-prompt');
        if (promptArea && p.text_prompt) promptArea.value = p.text_prompt;

        const initialAuthor = document.getElementById('initial-agent-author');
        if (initialAuthor && !this.isConnected) initialAuthor.innerText = 'Aarav';
      }

      async saveActiveAgent() {
        const nameInput = document.getElementById('cfg-agent-name');
        const promptArea = document.getElementById('cfg-system-prompt');
        const headerSaveBtn = document.getElementById('btn-header-save');
        const headerSaveText = document.getElementById('btn-header-save-text');

        const updatedPersona = {
          id: 'indian_pro',
          name: (nameInput ? nameInput.value.trim() : '') || 'Aarav (Indian English • Articulate & Natural)',
          description: 'Natural, articulate, and crystal-clear Indian English conversational AI voice agent.',
          accent: 'Indian English',
          character: 'Professional',
          voice_prompt: 'NATM0.pt',
          neural_voice: 'en-IN-PrabhatNeural',
          text_prompt: promptArea ? promptArea.value.trim() : '',
        };

        try {
          const res = await fetch('/v1/agents/indian_pro', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedPersona)
          });
          const result = await res.json();
          if (result.agent) {
            const idx = this.personas.findIndex(p => p.id === 'indian_pro');
            if (idx >= 0) this.personas[idx] = result.agent;
            else this.personas.push(result.agent);

            this.activeAgentName = 'Aarav';
            this.activeNeuralVoice = 'en-IN-PrabhatNeural';

            // Visual feedback on Top-Right Header Save Button with color change
            if (headerSaveBtn) {
              headerSaveBtn.classList.add('saved');
              if (headerSaveText) headerSaveText.innerText = '✓ Saved!';
              setTimeout(() => {
                headerSaveBtn.classList.remove('saved');
                if (headerSaveText) headerSaveText.innerText = 'Save Agent';
              }, 2500);
            }

            const feedback = document.getElementById('agent-save-feedback');
            if (feedback) {
              feedback.innerText = 'Saved & Active';
              feedback.classList.add('show');
              setTimeout(() => feedback.classList.remove('show'), 2500);
            }
          }
        } catch (e) {
          alert('Failed to save agent: ' + e.message);
        }
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
          // Immediate barge-in cutoff & start fresh turn
          this.currentAgentBubble = null;
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
          this.currentAgentBubble = null;
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

        // Start speech recognition locked to Indian English
        this.recognizer.setLang('en-IN');
        this.recognizer.start();

        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${window.location.host}/v1/realtime?persona_id=${encodeURIComponent(this.activePersona)}&neural_voice=${encodeURIComponent(this.activeNeuralVoice || 'en-IN-PrabhatNeural')}&voice_prompt=${encodeURIComponent(this.activeVoice || 'NATM0.pt')}&accent=Indian%20English&character=${encodeURIComponent(this.activeCharacter || 'Professional')}`;
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

        this.currentAgentBubble = null;
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
        if (author === 'user') {
          this.currentAgentBubble = null;
        }
      }

      appendTranscriptToken(author, token) {
        const transcript = document.getElementById('chat-transcript');
        if (author === 'agent') {
          if (!this.currentAgentBubble) {
            const row = document.createElement('div');
            row.className = 'message-row agent';
            const authorName = this.activeAgentName || 'Agent';
            row.innerHTML = `
              <span class="message-author">${authorName}</span>
              <div class="bubble"></div>
            `;
            transcript.appendChild(row);
            this.currentAgentBubble = row.querySelector('.bubble');
          }
          this.currentAgentBubble.innerText += token;
          transcript.scrollTop = transcript.scrollHeight;
        }
      }

      disconnect() {
        this.currentAgentBubble = null;
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
