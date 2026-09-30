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

    /* Voice Clone Modal & Dialog Styles */
    .modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.78);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease-out;
    }
    .modal-backdrop.active {
      opacity: 1;
      pointer-events: auto;
    }
    .modal-card {
      width: 92%;
      max-width: 520px;
      max-height: 88vh;
      background: #1c1c1e;
      border: 0.5px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.9), 0 0 1px 1px rgba(255, 255, 255, 0.1);
      transform: scale(0.96);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      overflow-y: auto;
    }
    .modal-backdrop.active .modal-card {
      transform: scale(1);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 0.5px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 12px;
    }
    .modal-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #fff;
    }
    .btn-modal-close {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: var(--text-secondary);
      width: 28px;
      height: 28px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .btn-modal-close:hover {
      background: rgba(255, 255, 255, 0.2);
      color: #fff;
    }
    .clone-tabs {
      display: flex;
      background: rgba(0, 0, 0, 0.4);
      padding: 3px;
      border-radius: 10px;
      border: 0.5px solid var(--border-subtle);
      gap: 4px;
    }
    .clone-tab-btn {
      flex: 1;
      padding: 7px 12px;
      border: none;
      background: transparent;
      color: var(--text-secondary);
      font-size: 12px;
      font-weight: 600;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .clone-tab-btn.active {
      background: rgba(255, 255, 255, 0.15);
      color: #fff;
    }
    .rec-box {
      border: 1px dashed rgba(255, 255, 255, 0.22);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      background: rgba(0, 0, 0, 0.25);
      text-align: center;
    }
    .btn-record-circle {
      width: 58px;
      height: 58px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--apple-red), #e0245e);
      border: none;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: 0 4px 16px rgba(255, 59, 48, 0.4);
    }
    .btn-record-circle.recording {
      animation: pulseRecord 1.2s infinite;
      background: #ff3b30;
    }
    @keyframes pulseRecord {
      0% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.7); }
      70% { box-shadow: 0 0 0 14px rgba(255, 59, 48, 0); }
      100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
    }
    .cloned-voice-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.04);
      border: 0.5px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px;
      margin-bottom: 6px;
    }
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

      <!-- Colloquial Indian English Voice Selector & Voice Cloner -->
      <div class="segmented-control" style="margin-bottom: 4px;">
        <button id="btn-agent-aarav" class="segmented-btn active" onclick="window.studioApp.selectAgentPreset('indian_pro')">Aarav (Male)</button>
        <button id="btn-agent-priya" class="segmented-btn" onclick="window.studioApp.selectAgentPreset('indian_priya')">Priya (Female)</button>
        <button id="btn-agent-clone" class="segmented-btn" style="color: var(--apple-purple); font-weight: 600;" onclick="window.studioApp.openCloneModal()">🧬 Clone Voice</button>
      </div>

      <!-- Colloquial Voice Profile Card -->
      <div class="single-agent-profile">
        <div class="profile-top">
          <span class="profile-name" id="profile-agent-name">Aarav</span>
          <span class="profile-badge" id="profile-agent-badge">Colloquial Kokoro Neural (Male)</span>
        </div>
        <div class="profile-desc" id="profile-agent-desc">Natural, articulate, and colloquial Indian English conversational voice agent.</div>
        <div class="profile-meta-row">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
          <span id="profile-agent-meta">aarav_colloquial (Kokoro 24 kHz)</span>
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
          <input type="text" id="cfg-agent-name" class="form-input" value="Aarav (Colloquial Indian English • Male)" placeholder="Agent Name">
        </div>

        <div class="form-group">
          <label class="form-label" for="cfg-call-flow">Call Flow &amp; Role</label>
          <select id="cfg-call-flow" class="form-select" onchange="window.studioApp.onCallFlowChange()">
            <option value="conversational_companion">Conversational Companion (Warm &amp; Witty)</option>
            <option value="customer_support">Customer Support &amp; Resolution</option>
            <option value="tech_specialist">Tech &amp; AI Specialist</option>
            <option value="inbound_concierge">Inbound Concierge &amp; Booking</option>
          </select>
        </div>

        <div class="form-row">
          <div class="form-group flex-1">
            <label class="form-label" for="cfg-voice-tone">Voice Tone</label>
            <select id="cfg-voice-tone" class="form-select" onchange="window.studioApp.onVoiceToneChange()">
              <option value="Natural &amp; Articulate">Natural &amp; Articulate</option>
              <option value="Warm &amp; Empathetic">Warm &amp; Empathetic</option>
              <option value="Energetic &amp; Witty">Energetic &amp; Witty</option>
              <option value="Professional &amp; Calm">Professional &amp; Calm</option>
              <option value="Casual &amp; Direct">Casual &amp; Direct</option>
            </select>
          </div>
          <div class="form-group flex-1">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <label class="form-label" for="cfg-neural-voice">Neural Voice</label>
              <a href="javascript:void(0)" onclick="window.studioApp.openCloneModal()" style="font-size:10px; color:var(--apple-purple); text-decoration:none; font-weight:600;">+ Clone</a>
            </div>
            <select id="cfg-neural-voice" class="form-select" onchange="window.studioApp.onNeuralVoiceChange()" style="font-family: var(--font-mono); font-size: 11px;">
              <optgroup label="🇮🇳 Indian English (Colloquial)">
                <option value="aarav_colloquial">aarav_colloquial (Male)</option>
                <option value="priya_colloquial">priya_colloquial (Female)</option>
                <option value="kabir">kabir (Deep Male)</option>
                <option value="ananya">ananya (Expressive Female)</option>
              </optgroup>
              <optgroup label="🌍 Global English">
                <option value="am_adam">am_adam (Dynamic Male)</option>
                <option value="af_bella">af_bella (Warm Female)</option>
                <option value="af_sarah">af_sarah (Professional Female)</option>
                <option value="am_michael">am_michael (Deep Baritone)</option>
                <option value="af_nova">af_nova (Bright &amp; Lively)</option>
              </optgroup>
              <optgroup id="optgroup-cloned-voices" label="🧬 Cloned Voices">
                <!-- Populated dynamically via /v1/voices/cloned -->
              </optgroup>
            </select>
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

      <!-- Transcript Container — starts empty; agent greeting appears when call begins -->
      <div class="chat-transcript" id="chat-transcript">
      </div>


      <!-- Bottom Call Controls -->
      <div class="call-controls">
        <button id="btn-mic-toggle" class="btn-circle" title="Toggle Mute" onclick="window.studioApp.toggleMute()">
          <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
          </svg>
        </button>

        <button id="btn-ptt-toggle" class="btn-circle" title="Toggle Push-to-Talk (Hold Space)" onclick="window.studioApp.togglePttMode()" style="font-size: 11px; font-weight: 700;">PTT</button>

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
          <span class="telemetry-val" id="active-neural-voice-badge">aarav_colloquial</span>
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
          <span class="telemetry-label">Frames In / Out</span>
          <span class="telemetry-val"><span id="telemetry-frames-in">0</span> / <span id="telemetry-frames-out">0</span></span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">STT Latency</span>
          <span class="telemetry-val" id="telemetry-stt-ms">--</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">LLM TTFT</span>
          <span class="telemetry-val" id="telemetry-llm-ttft">--</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">TTS TTFA</span>
          <span class="telemetry-val" id="telemetry-tts-ttfa">--</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Total TTFA</span>
          <span class="telemetry-val" id="telemetry-total-ttfa" style="color: var(--apple-green);">--</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Barge-in Events</span>
          <span class="telemetry-val" id="telemetry-barge-in">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Audio Cleaner</span>
          <button id="btn-toggle-bypass" onclick="window.studioApp.toggleAudioBypass()" class="segmented-btn active" style="font-size: 10px; padding: 2px 6px;">Clean: ON</button>
        </div>
        <div class="telemetry-row" style="margin-top: 4px; gap: 4px;">
          <button onclick="window.studioApp.downloadRawWav()" class="segmented-btn" style="font-size: 10px; flex: 1;">Raw WAV</button>
          <button onclick="window.studioApp.downloadCleanWav()" class="segmented-btn" style="font-size: 10px; flex: 1;">Clean WAV</button>
        </div>
      </div>

      <!-- Live Pipeline Stages & GPU Status Panel -->
      <div class="panel-header" style="margin-top: 8px;">
        <span class="panel-title">Pipeline Stage Status &amp; GPU</span>
        <span class="doc-badge" id="gpu-device-badge" style="color: var(--apple-cyan);">CUDA</span>
      </div>
      <div class="telemetry-group" style="padding: 10px; font-size: 11px;">
        <div class="telemetry-row">
          <span class="telemetry-label">🎤 Heard Speech</span>
          <span class="telemetry-val" id="stage-heard-speech">Idle</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">📝 STT Transcript</span>
          <span class="telemetry-val" id="stage-transcript">--</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">🧠 LLM Thinking</span>
          <span class="telemetry-val" id="stage-llm-thinking">Idle</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">🔊 TTS Speaking</span>
          <span class="telemetry-val" id="stage-tts-speaking">Idle</span>
        </div>
        <div class="telemetry-row" style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 4px; margin-top: 4px;">
          <span class="telemetry-label">⚡ GPU VRAM</span>
          <span class="telemetry-val" id="gpu-vram-text">-- / 4096 MB</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">🔥 GPU Temp / Util</span>
          <span class="telemetry-val" id="gpu-temp-util">--°C / --%</span>
        </div>
      </div>
    </div>

  </div>

  <!-- Apple Intelligence Inspired Voice Cloning Modal -->
  <div id="voice-clone-modal" class="modal-backdrop">
    <div class="modal-card">
      <div class="modal-header">
        <div class="modal-title">
          <svg width="20" height="20" fill="var(--apple-purple)" viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
          Voice Cloning Studio
        </div>
        <button class="btn-modal-close" onclick="window.studioApp.closeCloneModal()">✕</button>
      </div>

      <div class="clone-tabs">
        <button id="btn-tab-record" class="clone-tab-btn active" onclick="window.studioApp.switchCloneTab('record')">🎙️ Record Microphone (8s)</button>
        <button id="btn-tab-upload" class="clone-tab-btn" onclick="window.studioApp.switchCloneTab('upload')">📁 Upload Audio File</button>
      </div>

      <!-- Tab 1: Record Directly from Microphone -->
      <div id="clone-tab-record">
        <div class="rec-box">
          <div style="font-size: 13px; font-weight: 600; color: #fff;">Acoustic Voice Calibration</div>
          <div style="font-size: 11.5px; color: var(--text-secondary); max-width: 380px; line-height: 1.4;">
            Read this sentence aloud in your natural speaking voice:
            <div style="margin-top: 6px; padding: 8px 10px; background: rgba(255,255,255,0.06); border-radius: 8px; color: var(--apple-cyan); font-style: italic;">
              "Hi, I am recording my voice sample for this real-time conversational AI. Notice my tone, rhythm, and natural accent."
            </div>
          </div>

          <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; margin-top: 4px;">
            <button id="btn-start-record" class="btn-record-circle" onclick="window.studioApp.toggleCloneRecord()">
              <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
            </button>
            <span id="clone-timer-text" style="font-family: var(--font-mono); font-size: 13px; font-weight: 700; color: var(--apple-red);">00:08</span>
            <span id="rec-prompt-status" style="font-size: 11px; color: var(--text-secondary);">Click red button to start 8-second recording</span>
          </div>

          <audio id="clone-audio-preview" controls style="display:none; width: 100%; height: 32px; margin-top: 6px;"></audio>
        </div>
      </div>

      <!-- Tab 2: Upload Audio File -->
      <div id="clone-tab-upload" style="display: none;">
        <div class="rec-box" onclick="document.getElementById('clone-file-input').click()" style="cursor: pointer;">
          <input type="file" id="clone-file-input" accept="audio/*,.wav,.mp3,.m4a,.ogg" style="display: none;" onchange="window.studioApp.onCloneFileUpload(event)">
          <svg width="32" height="32" fill="var(--text-secondary)" viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>
          <div style="font-size: 12.5px; font-weight: 600; color: #fff;">Click or Drag &amp; Drop Audio File</div>
          <div style="font-size: 11px; color: var(--text-secondary);">Supports WAV, MP3, M4A, OGG (3–30 seconds recommended)</div>
          <div id="upload-file-name" style="font-family: var(--font-mono); font-size: 11.5px; color: var(--apple-cyan); margin-top: 4px;"></div>
          <audio id="upload-audio-preview" controls style="display:none; width: 100%; height: 32px; margin-top: 6px;"></audio>
        </div>
      </div>

      <!-- Metadata Fields -->
      <div style="display: flex; gap: 8px;">
        <div class="form-group flex-1">
          <label class="form-label" for="clone-voice-name">Voice Profile Name</label>
          <input type="text" id="clone-voice-name" class="form-input" placeholder="e.g. My Voice, Rohan, Sneha" value="My Voice">
        </div>
        <div class="form-group flex-1">
          <label class="form-label" for="clone-voice-gender">Gender Tone Bias</label>
          <select id="clone-voice-gender" class="form-select">
            <option value="auto">Auto-detect from acoustics</option>
            <option value="male">Male (Full / Baritone)</option>
            <option value="female">Female (Warm / Bright)</option>
          </select>
        </div>
      </div>

      <button id="btn-submit-clone" class="btn-save-agent" style="background: linear-gradient(135deg, var(--apple-purple), #9d4edd); width: 100%; padding: 10px; font-size: 13px;" onclick="window.studioApp.submitVoiceClone()">
        🧬 Create &amp; Activate Cloned Voice
      </button>

      <!-- Manage Existing Clones -->
      <div style="border-top: 0.5px solid rgba(255, 255, 255, 0.1); padding-top: 10px;">
        <div style="font-size: 11px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 6px;">Saved Cloned Voices</div>
        <div id="cloned-voices-list" style="max-height: 120px; overflow-y: auto;">
          <!-- Populated dynamically -->
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
        this.activeSources = [];
        this.onFrameCallback = null;
        this.onEnergyCallback = null;
      }

      async initialize() {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!this.audioCtx) {
          try {
            this.audioCtx = new AudioContextClass();
          } catch (e) {
            console.error('AudioContext creation error:', e);
          }
        }
        if (this.audioCtx && this.audioCtx.state === 'suspended') {
          try {
            await this.audioCtx.resume();
          } catch (e) {
            console.warn('AudioContext resume error:', e);
          }
        }
        if (this.audioCtx) {
          this.nextPlayTime = this.audioCtx.currentTime;
        }
      }

      async startMicrophone(onFrame, onEnergy) {
        this.onFrameCallback = onFrame;
        this.onEnergyCallback = onEnergy;
        await this.initialize();

        if (this.audioCtx && this.audioCtx.state === 'suspended') {
          try { await this.audioCtx.resume(); } catch (e) {}
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

      async playChunk(floatSamples) {
        if (!this.audioCtx) {
          try { await this.initialize(); } catch (e) {}
        }
        if (this.audioCtx && this.audioCtx.state === 'suspended') {
          try { await this.audioCtx.resume(); } catch (e) {}
        }
        if (!this.audioCtx) return;

        // Skip pure silence frames so nextPlayTime doesn't wander into future
        let sumSq = 0;
        for (let i = 0; i < floatSamples.length; i++) {
          sumSq += floatSamples[i] * floatSamples[i];
        }
        const rms = Math.sqrt(sumSq / floatSamples.length);
        if (rms < 0.0004) return;

        const now = this.audioCtx.currentTime;
        if (this.nextPlayTime < now || this.nextPlayTime > now + 0.35) {
          // Snap directly to current time with 15ms buffer for immediate speech
          this.nextPlayTime = now + 0.015;
        }

        const buffer = this.audioCtx.createBuffer(1, floatSamples.length, this.sampleRate);
        buffer.copyToChannel(floatSamples, 0);

        const source = this.audioCtx.createBufferSource();
        source.buffer = buffer;

        // Dedicated gain node for boosted, clear agent voice output
        const gainNode = this.audioCtx.createGain();
        gainNode.gain.value = 1.35;
        source.connect(gainNode);
        gainNode.connect(this.audioCtx.destination);

        source.start(this.nextPlayTime);
        this.nextPlayTime += buffer.duration;

        this.activeSources.push(source);
        source.onended = () => {
          const idx = this.activeSources.indexOf(source);
          if (idx >= 0) this.activeSources.splice(idx, 1);
        };
      }

      stopPlayback() {
        if (this.activeSources && this.activeSources.length > 0) {
          for (const s of this.activeSources) {
            try { s.stop(); s.disconnect(); } catch (e) {}
          }
          this.activeSources = [];
        }
        if (this.audioCtx) {
          this.nextPlayTime = this.audioCtx.currentTime;
        }
      }

      toggleMute() {
        this.isMuted = !this.isMuted;
        return this.isMuted;
      }

      stop() {
        this.stopPlayback();
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
            const floatSamples = new Float32Array(event.data.slice(1));
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
        this.activeCallFlow = 'conversational_companion';
        this.isConnected = false;
        this.agentSpeaking = false;
        this.currentVoiceFilter = 'All';
        this.currentAgentBubble = null;

        this.framesIn = 0;
        this.framesOut = 0;
        this.bargeIns = 0;
        this.isPttMode = false;

        this.personas = [];
        this.voices = [];
        this.promptManager = new SystemPromptManager();

        this.recognizer = new SpeechRecognizer(
          (text, isFinal) => this.handleSpeechRecognized(text, isFinal),
          (err) => console.log('Recognizer info:', err)
        );

        this.initElements();
        this.initSocketEvents();
        this.setupPttListeners();
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
          const elOut = document.getElementById('telemetry-frames-out');
          if (elOut) elOut.innerText = this.framesOut;

          let sumSq = 0;
          for (let i = 0; i < floatSamples.length; i++) {
            sumSq += floatSamples[i] * floatSamples[i];
          }
          const rms = Math.sqrt(sumSq / floatSamples.length);
          if (rms > 0.005) {
            this.agentSpeaking = true;
            this.orb.setAgentSpeaking(true);
            const spk = document.getElementById('speaker-status');
            if (spk) spk.innerText = `${this.activeAgentName} Speaking`;
            const stTTS = document.getElementById('stage-tts-speaking');
            if (stTTS) stTTS.innerText = 'Speaking (24kHz)';
          } else if (this.agentSpeaking && (!this.audio.activeSources || this.audio.activeSources.length === 0)) {
            this.agentSpeaking = false;
            this.orb.setAgentSpeaking(false);
            const spk = document.getElementById('speaker-status');
            if (spk) spk.innerText = `Listening (${this.activeAgentName} ready)`;
            const stTTS = document.getElementById('stage-tts-speaking');
            if (stTTS) stTTS.innerText = 'Idle';
          }
          this.audio.playChunk(floatSamples);
        };

        this.socket.onText = (token) => {
          this.appendTranscriptToken('agent', token);
        };

        this.socket.onMetadata = (meta) => {
          if (meta.event === 'session_started') {
            this.activeSessionId = meta.session_id;
          }
          if (meta.event === 'user_transcript' && meta.text) {
            this.updateUserLiveBubble(meta.text, true);
            const stageTr = document.getElementById('stage-transcript');
            if (stageTr) stageTr.innerText = meta.text.slice(0, 32);
          }
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
        this.activeNeuralVoice = 'aarav_colloquial';
        this.activeVoice = 'NATM0.pt';
        this.activeCharacter = 'Professional';
        this.clonedVoices = [];
        this.startMetricsPolling();
        await this.loadPersonas();
        await this.loadClonedVoices();
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

      async loadClonedVoices() {
        try {
          const res = await fetch('/v1/voices/cloned');
          const data = await res.json();
          this.clonedVoices = data.cloned_voices || [];
          this.populateClonedVoiceOptions();
        } catch (e) {
          console.warn('Failed loading cloned voices:', e);
        }
      }

      populateClonedVoiceOptions() {
        const optgroup = document.getElementById('optgroup-cloned-voices');
        if (!optgroup) return;
        optgroup.innerHTML = '';
        if (!this.clonedVoices || this.clonedVoices.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.disabled = true;
          opt.innerText = 'No cloned voices yet (Click + Clone)';
          optgroup.appendChild(opt);
          return;
        }
        for (const cv of this.clonedVoices) {
          const opt = document.createElement('option');
          opt.value = cv.id;
          opt.innerText = `🧬 ${cv.name} (${cv.gender || 'Custom'})`;
          optgroup.appendChild(opt);
        }
      }

      selectAgentPreset(presetId) {
        this.activePersona = presetId || 'indian_pro';
        const btnAarav = document.getElementById('btn-agent-aarav');
        const btnPriya = document.getElementById('btn-agent-priya');
        if (btnAarav) btnAarav.classList.toggle('active', this.activePersona === 'indian_pro');
        if (btnPriya) btnPriya.classList.toggle('active', this.activePersona === 'indian_priya');
        this.populateCustomizer();
      }

      populateCustomizer() {
        const isPriya = this.activePersona === 'indian_priya';
        let p = this.personas.find(item => item.id === this.activePersona);
        if (!p) {
          if (isPriya) {
            p = {
              id: 'indian_priya',
              name: 'Priya (Colloquial Indian English • Female)',
              character: 'Warm',
              neural_voice: 'priya_colloquial',
              voice_prompt: 'NATF0.pt',
              description: 'Warm, bright, empathetic, and colloquial Indian English speaker.',
              text_prompt: "You are Priya, a warm, articulate Indian English voice assistant.\nAlways speak with a natural, colloquial Indian English cadence.\nUse short, conversational 1-2 sentence replies with natural idioms like 'Haanji', 'Sure thing', 'Tell me'.",
            };
          } else {
            p = {
              id: 'indian_pro',
              name: 'Aarav (Colloquial Indian English • Male)',
              character: 'Professional',
              neural_voice: 'aarav_colloquial',
              voice_prompt: 'NATM0.pt',
              description: 'Articulate, natural, conversational Indian English speaker with relaxed cadence.',
              text_prompt: "You are Aarav, an articulate Indian English voice assistant.\nAlways speak with a natural, colloquial Indian English cadence.\nUse short, conversational 1-2 sentence replies with natural idioms like 'Haanji', 'Got it', 'Tell me'.",
            };
          }
        }

        this.activeAgentName = isPriya ? 'Priya' : 'Aarav';
        this.activeCharacter = p.character || (isPriya ? 'Warm' : 'Professional');
        this.activeNeuralVoice = isPriya ? 'priya_colloquial' : (p.neural_voice || 'aarav_colloquial');
        this.activeVoice = isPriya ? 'NATF0.pt' : (p.voice_prompt || 'NATM0.pt');

        const badge = document.getElementById('active-agent-badge');
        if (badge) badge.innerText = this.activeAgentName;

        const profileName = document.getElementById('profile-agent-name');
        if (profileName) profileName.innerText = this.activeAgentName;

        const profileDesc = document.getElementById('profile-agent-desc');
        if (profileDesc) profileDesc.innerText = p.description || `${this.activeAgentName} • Colloquial Indian English Voice.`;

        const profileMeta = document.getElementById('profile-agent-meta');
        if (profileMeta) profileMeta.innerText = `${this.activeNeuralVoice} (Kokoro 24 kHz)`;

        const neuralBadge = document.getElementById('active-neural-voice-badge');
        if (neuralBadge) neuralBadge.innerText = this.activeNeuralVoice;

        const profileBadge = document.getElementById('profile-agent-badge');
        if (profileBadge) profileBadge.innerText = isPriya ? 'Colloquial Kokoro Neural (Female)' : 'Colloquial Kokoro Neural (Male)';

        const toneSelect = document.getElementById('cfg-voice-tone');
        if (toneSelect) toneSelect.value = isPriya ? 'Warm & Empathetic' : 'Natural & Articulate';

        const nameInput = document.getElementById('cfg-agent-name');
        if (nameInput) nameInput.value = p.name || `${this.activeAgentName} (Colloquial Indian English)`;

        const voiceSelect = document.getElementById('cfg-neural-voice');
        if (voiceSelect) voiceSelect.value = this.activeNeuralVoice;

        const callFlowSelect = document.getElementById('cfg-call-flow');
        if (callFlowSelect) callFlowSelect.value = this.activeCallFlow;

        const promptArea = document.getElementById('cfg-system-prompt');
        if (promptArea) {
          promptArea.value = this.getCallFlowPrompt(this.activeAgentName, this.activeCallFlow);
        }

        const initialAuthor = document.getElementById('initial-agent-author');
        if (initialAuthor && !this.isConnected) initialAuthor.innerText = this.activeAgentName;

        const speakerStatus = document.getElementById('speaker-status');
        if (speakerStatus && !this.isConnected) {
          speakerStatus.innerText = `Ready to Speak (${this.activeAgentName})`;
        }

        const callBtnText = document.getElementById('call-btn-text');
        if (callBtnText && !this.isConnected) {
          callBtnText.innerText = `Start Call with ${this.activeAgentName}`;
        }
      }

      onNeuralVoiceChange() {
        const select = document.getElementById('cfg-neural-voice');
        if (!select) return;
        const val = select.value;
        if (!val) return;
        this.activeNeuralVoice = val;

        // Check if cloned voice
        const cloned = (this.clonedVoices || []).find(v => v.id === val);
        if (cloned) {
          this.activeAgentName = cloned.name;
          this.activeCharacter = cloned.gender === 'Female' ? 'Warm & Empathetic' : 'Natural & Articulate';
          const nameInput = document.getElementById('cfg-agent-name');
          if (nameInput) nameInput.value = `${cloned.name} (Cloned Voice)`;
          const toneSelect = document.getElementById('cfg-voice-tone');
          if (toneSelect) toneSelect.value = this.activeCharacter;
          const badge = document.getElementById('active-agent-badge');
          if (badge) badge.innerText = cloned.name;
          const profileName = document.getElementById('profile-agent-name');
          if (profileName) profileName.innerText = cloned.name;
          const profileBadge = document.getElementById('profile-agent-badge');
          if (profileBadge) profileBadge.innerText = `Cloned Neural Voice (${cloned.gender || 'Custom'})`;
          const profileDesc = document.getElementById('profile-agent-desc');
          if (profileDesc) profileDesc.innerText = `Custom zero-shot neural clone (${cloned.f0_pitch ? Math.round(cloned.f0_pitch) + 'Hz' : 'Acoustic Latent'}).`;
          const profileMeta = document.getElementById('profile-agent-meta');
          if (profileMeta) profileMeta.innerText = `${val} (Kokoro 24 kHz)`;
          const speakerStatus = document.getElementById('speaker-status');
          if (speakerStatus && !this.isConnected) speakerStatus.innerText = `Ready to Speak (${cloned.name})`;
          const callBtnText = document.getElementById('call-btn-text');
          if (callBtnText && !this.isConnected) callBtnText.innerText = `Start Call with ${cloned.name}`;
          const promptArea = document.getElementById('cfg-system-prompt');
          if (promptArea) promptArea.value = this.getCallFlowPrompt(cloned.name, this.activeCallFlow);
          return;
        }

        const isPriya = val.includes('priya') || val.includes('ananya') || val.startsWith('af_') || val.startsWith('hf_');
        this.activeAgentName = isPriya ? 'Priya' : 'Aarav';
        this.activeCharacter = isPriya ? 'Warm & Empathetic' : 'Natural & Articulate';
        
        const toneSelect = document.getElementById('cfg-voice-tone');
        if (toneSelect) toneSelect.value = this.activeCharacter;
        const nameInput = document.getElementById('cfg-agent-name');
        if (nameInput) nameInput.value = `${this.activeAgentName} (Colloquial Indian English)`;

        const badge = document.getElementById('active-agent-badge');
        if (badge) badge.innerText = this.activeAgentName;
        const profileName = document.getElementById('profile-agent-name');
        if (profileName) profileName.innerText = this.activeAgentName;
        const profileBadge = document.getElementById('profile-agent-badge');
        if (profileBadge) profileBadge.innerText = isPriya ? 'Neural Voice (Female)' : 'Neural Voice (Male)';
        const profileMeta = document.getElementById('profile-agent-meta');
        if (profileMeta) profileMeta.innerText = `${val} (Kokoro 24 kHz)`;

        const btnAarav = document.getElementById('btn-agent-aarav');
        const btnPriya = document.getElementById('btn-agent-priya');
        if (btnAarav) btnAarav.classList.toggle('active', !isPriya);
        if (btnPriya) btnPriya.classList.toggle('active', isPriya);

        const speakerStatus = document.getElementById('speaker-status');
        if (speakerStatus && !this.isConnected) speakerStatus.innerText = `Ready to Speak (${this.activeAgentName})`;
        const callBtnText = document.getElementById('call-btn-text');
        if (callBtnText && !this.isConnected) callBtnText.innerText = `Start Call with ${this.activeAgentName}`;
        const promptArea = document.getElementById('cfg-system-prompt');
        if (promptArea) promptArea.value = this.getCallFlowPrompt(this.activeAgentName, this.activeCallFlow);
      }

      onVoiceToneChange() {
        const select = document.getElementById('cfg-voice-tone');
        if (!select) return;
        this.activeCharacter = select.value;
      }

      // Voice Cloning Modal Controllers
      openCloneModal() {
        const modal = document.getElementById('voice-clone-modal');
        if (modal) modal.classList.add('active');
        this.renderClonedVoicesManager();
      }

      closeCloneModal() {
        const modal = document.getElementById('voice-clone-modal');
        if (modal) modal.classList.remove('active');
        this.stopCloneRecord(true);
      }

      switchCloneTab(tab) {
        document.getElementById('clone-tab-record').style.display = tab === 'record' ? 'block' : 'none';
        document.getElementById('clone-tab-upload').style.display = tab === 'upload' ? 'block' : 'none';
        document.getElementById('btn-tab-record').classList.toggle('active', tab === 'record');
        document.getElementById('btn-tab-upload').classList.toggle('active', tab === 'upload');
      }

      async toggleCloneRecord() {
        const btn = document.getElementById('btn-start-record');
        if (btn.classList.contains('recording')) {
          this.stopCloneRecord();
        } else {
          await this.startCloneRecord();
        }
      }

      async startCloneRecord() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          this.cloneMediaRecorder = new MediaRecorder(stream);
          this.cloneAudioChunks = [];
          this.cloneMediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.cloneAudioChunks.push(e.data);
          };

          this.cloneMediaRecorder.onstop = () => {
            const mime = (this.cloneMediaRecorder && this.cloneMediaRecorder.mimeType) || 'audio/webm';
            const blob = new Blob(this.cloneAudioChunks, { type: mime });
            this.recordedCloneBlob = blob;
            const audioUrl = URL.createObjectURL(blob);
            const preview = document.getElementById('clone-audio-preview');
            preview.src = audioUrl;
            preview.style.display = 'block';
            document.getElementById('btn-submit-clone').disabled = false;
            document.getElementById('rec-prompt-status').innerText = 'Audio recorded! Click button below to create clone.';
            stream.getTracks().forEach(t => t.stop());
          };

          this.cloneMediaRecorder.start();
          document.getElementById('btn-start-record').classList.add('recording');
          document.getElementById('rec-prompt-status').innerText = 'Recording... Speak clearly into microphone.';

          let timeLeft = 8;
          const timerEl = document.getElementById('clone-timer-text');
          timerEl.innerText = `00:0${timeLeft}`;
          clearInterval(this.cloneTimerInterval);
          this.cloneTimerInterval = setInterval(() => {
            timeLeft--;
            timerEl.innerText = `00:0${timeLeft}`;
            if (timeLeft <= 0) {
              clearInterval(this.cloneTimerInterval);
              this.stopCloneRecord();
            }
          }, 1000);
        } catch (e) {
          alert('Microphone access required for voice cloning: ' + e.message);
        }
      }

      stopCloneRecord(cancel = false) {
        clearInterval(this.cloneTimerInterval);
        const btn = document.getElementById('btn-start-record');
        if (btn) btn.classList.remove('recording');
        if (this.cloneMediaRecorder && this.cloneMediaRecorder.state === 'recording') {
          this.cloneMediaRecorder.stop();
        }
        if (cancel) {
          this.recordedCloneBlob = null;
        }
      }

      onCloneFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        this.uploadedCloneFile = file;
        const audioUrl = URL.createObjectURL(file);
        const preview = document.getElementById('upload-audio-preview');
        preview.src = audioUrl;
        preview.style.display = 'block';
        document.getElementById('btn-submit-clone').disabled = false;
        document.getElementById('upload-file-name').innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;
      }

      async submitVoiceClone() {
        const nameInput = document.getElementById('clone-voice-name');
        const voiceName = (nameInput ? nameInput.value.trim() : '') || 'My Voice';
        const genderSelect = document.getElementById('clone-voice-gender');
        const gender = genderSelect ? genderSelect.value : 'auto';

        let audioFile = this.recordedCloneBlob || this.uploadedCloneFile;
        if (!audioFile) {
          alert('Please record microphone audio or select an audio file first.');
          return;
        }

        const submitBtn = document.getElementById('btn-submit-clone');
        submitBtn.disabled = true;
        submitBtn.innerText = 'Extracting Acoustic Vectors...';

        const fileName = audioFile.name || (audioFile.type && audioFile.type.includes('webm') ? 'recorded_sample.webm' : 'recorded_sample.wav');
        const formData = new FormData();
        formData.append('audio', audioFile, fileName);
        formData.append('voice_name', voiceName);
        if (gender !== 'auto') formData.append('gender', gender);

        try {
          const res = await fetch('/v1/voices/clone', {
            method: 'POST',
            body: formData,
          });
          const data = await res.json();
          if (data.status === 'cloned') {
            await this.loadClonedVoices();
            // Automatically select the new cloned voice!
            const select = document.getElementById('cfg-neural-voice');
            if (select) {
              select.value = data.voice.id;
              this.onNeuralVoiceChange();
            }
            this.closeCloneModal();
            const feedback = document.getElementById('agent-save-feedback');
            if (feedback) {
              feedback.innerText = `Voice Cloned: ${voiceName}!`;
              feedback.classList.add('show');
              setTimeout(() => feedback.classList.remove('show'), 3500);
            }
          } else {
            alert('Voice cloning failed: ' + (data.detail || JSON.stringify(data)));
          }
        } catch (e) {
          alert('Error during voice cloning: ' + e.message);
        } finally {
          submitBtn.disabled = false;
          submitBtn.innerText = '🧬 Create & Activate Cloned Voice';
        }
      }

      renderClonedVoicesManager() {
        const list = document.getElementById('cloned-voices-list');
        if (!list) return;
        list.innerHTML = '';
        if (!this.clonedVoices || this.clonedVoices.length === 0) {
          list.innerHTML = '<div style="font-size:12px; color:var(--text-secondary); text-align:center; padding:12px;">No custom cloned voices created yet.</div>';
          return;
        }
        for (const cv of this.clonedVoices) {
          const item = document.createElement('div');
          item.className = 'cloned-voice-item';
          item.innerHTML = `
            <div>
              <div style="font-weight:600; font-size:12.5px; color:#fff;">🧬 ${cv.name}</div>
              <div style="font-size:10.5px; color:var(--text-secondary); font-family:var(--font-mono);">${cv.gender} • Pitch: ${cv.f0_pitch ? Math.round(cv.f0_pitch) + ' Hz' : 'Auto'}</div>
            </div>
            <div style="display:flex; gap:6px;">
              <button class="segmented-btn" style="padding:4px 8px; font-size:11px;" onclick="window.studioApp.activateClonedVoice('${cv.id}')">Select</button>
              <button class="segmented-btn" style="padding:4px 8px; font-size:11px; color:var(--apple-red);" onclick="window.studioApp.deleteClonedVoice('${cv.id}')">✕</button>
            </div>
          `;
          list.appendChild(item);
        }
      }

      activateClonedVoice(voiceId) {
        const select = document.getElementById('cfg-neural-voice');
        if (select) {
          select.value = voiceId;
          this.onNeuralVoiceChange();
        }
        this.closeCloneModal();
      }

      async deleteClonedVoice(voiceId) {
        if (!confirm('Are you sure you want to delete this cloned voice?')) return;
        try {
          await fetch(`/v1/voices/cloned/${encodeURIComponent(voiceId)}`, { method: 'DELETE' });
          await this.loadClonedVoices();
          this.renderClonedVoicesManager();
        } catch (e) {
          alert('Failed to delete voice: ' + e.message);
        }
      }

      getCallFlowPrompt(agentName, role) {
        const prompts = {
          conversational_companion: `AGENT IDENTITY:\nYou are ${agentName}, a friendly, bright, and witty conversational partner from India chatting on a real-time voice call. You speak with natural warmth, relatable humor, and an authentic colloquial Indian English rhythm.\n\nCORE VOICE CONVERSATION RULES:\n1. Colloquial Indian English: Speak naturally with short sentences (10 to 18 words), natural contractions, and subtle discourse markers ('actually', 'na', 'simple, na?').\n2. Plain spoken text ONLY: NO markdown, NO asterisks, NO bullets, NO emojis.\n3. Zero robotic phrases: Never say 'How can I assist you today'.\n\nCALL FLOW STAGES:\n1. GREETING & PRESENCE: Acknowledge caller warmly.\n2. INTENT DISCOVERY: Listen to what caller brings up.\n3. CONCISE RESPONSE: 1 to 2 spoken sentences (under 30 words).\n4. CHECK-IN: Casual check ('Makes sense, na?').\n5. WARM WRAP-UP: End on a friendly note.`,
          customer_support: `AGENT IDENTITY:\nYou are ${agentName}, a helpful and empathetic customer resolution specialist from India. You speak polite, clear, colloquial Indian English. You remain calm, patient, and completely solution-oriented.\n\nCORE VOICE CONVERSATION RULES:\n1. Concise Spoken Delivery: 1 to 2 short sentences per turn.\n2. Ban robotic jargon: Address customer issues directly with natural human warmth.\n3. Plain spoken words only.\n\nCALL FLOW STAGES:\n1. GREETING: Warmly welcome caller and ask what issue they need help with.\n2. ISSUE CLARIFICATION: Acknowledge their situation with genuine care.\n3. DIRECT SOLUTION: Provide the fix or next step in 1–2 sentences.\n4. VERIFICATION: 'Does that solve it for you, or should we check anything else?'\n5. POLITE CLOSING: Wish them a wonderful day.`,
          tech_specialist: `AGENT IDENTITY:\nYou are ${agentName}, an articulate technology and AI engineer from India. You explain complex machine learning, software, and UPI/fintech concepts in simple, relatable conversational terms.\n\nCORE VOICE CONVERSATION RULES:\n1. Spoken intuition first: Explain core concept using a real-world analogy in 2 sentences.\n2. Plain spoken text only: NO markdown or bullet lists.\n3. Natural Indian English rhythm.\n\nCALL FLOW STAGES:\n1. GREETING: Connect with technical enthusiasm.\n2. CONCEPT INTUITION: 2-sentence relatable analogy.\n3. PRACTICAL APPLICATION: Practical example (UPI, Bangalore tech startups).\n4. DEPTH CHECK: Ask if they want technical depth or high-level.\n5. CONCLUDING INSIGHT: Clean summary.`,
          inbound_concierge: `AGENT IDENTITY:\nYou are ${agentName}, a gracious and organized front-desk concierge from India. You handle appointment scheduling, service inquiries, and reservations with prompt, friendly efficiency.\n\nCORE VOICE CONVERSATION RULES:\n1. Clear, organized spoken delivery.\n2. Plain spoken text only.\n3. One question at a time.\n\nCALL FLOW STAGES:\n1. WELCOME: Welcome caller cheerfully and offer assistance.\n2. NEED ASSESSMENT: Capture date, time, and service requirement step by step.\n3. CONFIRMATION: Read back key details crisply.\n4. NEXT STEP: 'I will send a confirmation SMS to your number, okay?'\n5. GRACEFUL SIGN-OFF: Thank them warmly.`
        };
        return prompts[role] || prompts.conversational_companion;
      }

      onCallFlowChange() {
        const select = document.getElementById('cfg-call-flow');
        if (!select) return;
        this.activeCallFlow = select.value;
        const promptArea = document.getElementById('cfg-system-prompt');
        if (promptArea) {
          promptArea.value = this.getCallFlowPrompt(this.activeAgentName, this.activeCallFlow);
        }
        const goalEl = document.getElementById('ctx-caller-goal');
        if (goalEl) {
          const titles = {
            conversational_companion: 'Conversational',
            customer_support: 'Support & Resolution',
            tech_specialist: 'Tech & AI Specialist',
            inbound_concierge: 'Booking & Concierge'
          };
          goalEl.innerText = titles[this.activeCallFlow] || 'Conversational';
        }
      }

      async saveActiveAgent() {
        const nameInput = document.getElementById('cfg-agent-name');
        const promptArea = document.getElementById('cfg-system-prompt');
        const headerSaveBtn = document.getElementById('btn-header-save');
        const headerSaveText = document.getElementById('btn-header-save-text');

        const updatedPersona = {
          id: this.activePersona,
          name: (nameInput ? nameInput.value.trim() : '') || `${this.activeAgentName} (Colloquial Indian English)`,
          description: `Natural, articulate, and colloquial Indian English conversational voice agent (${this.activeAgentName}).`,
          accent: 'Indian English',
          character: this.activeCharacter,
          voice_prompt: this.activeVoice,
          neural_voice: this.activeNeuralVoice,
          text_prompt: promptArea ? promptArea.value.trim() : '',
        };

        try {
          const res = await fetch(`/v1/agents/${encodeURIComponent(this.activePersona)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedPersona)
          });
          const result = await res.json();
          if (result.agent) {
            const idx = this.personas.findIndex(p => p.id === this.activePersona);
            if (idx >= 0) this.personas[idx] = result.agent;
            else this.personas.push(result.agent);

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

      togglePttMode() {
        this.isPttMode = !this.isPttMode;
        const btn = document.getElementById('btn-ptt-toggle');
        if (btn) {
          btn.style.background = this.isPttMode ? 'var(--apple-blue)' : 'var(--bg-pill)';
          btn.style.color = this.isPttMode ? '#fff' : 'var(--text-primary)';
          btn.innerText = this.isPttMode ? 'PTT ON' : 'PTT';
        }
        if (this.audio) {
          this.audio.isMuted = this.isPttMode;
        }
        const spk = document.getElementById('speaker-status');
        if (spk) spk.innerText = this.isPttMode ? 'Push & Hold Space to Speak' : `Listening (${this.activeAgentName} ready)`;
      }

      setupPttListeners() {
        let spacePressed = false;
        window.addEventListener('keydown', (e) => {
          if (!this.isPttMode || !this.isConnected) return;
          if (e.code === 'Space' && document.activeElement.id !== 'user-text-input' && !spacePressed) {
            spacePressed = true;
            if (this.audio) this.audio.isMuted = false;
            const spk = document.getElementById('speaker-status');
            if (spk) spk.innerText = 'Listening (Space Held)...';
            e.preventDefault();
          }
        });
        window.addEventListener('keyup', (e) => {
          if (!this.isPttMode || !this.isConnected) return;
          if (e.code === 'Space' && document.activeElement.id !== 'user-text-input') {
            spacePressed = false;
            if (this.audio) this.audio.isMuted = true;
            const spk = document.getElementById('speaker-status');
            if (spk) spk.innerText = 'Push & Hold Space to Speak';
            e.preventDefault();
          }
        });
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
        // Suppress laptop speaker acoustic echo loop while agent is speaking
        if (this.agentSpeaking) return;
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

        const stageTr = document.getElementById('stage-transcript');
        if (stageTr) stageTr.innerText = text.slice(0, 32);

        // Raw audio frames are streamed live to the backend GPU Faster-Whisper.
        // We do not send duplicate text messages here to prevent race conditions or echo cancellation issues.
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
              const elIn = document.getElementById('telemetry-frames-in');
              if (elIn) elIn.innerText = this.framesIn;
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
        const fallbackVoice = this.activePersona === 'indian_priya' ? 'priya_colloquial' : 'aarav_colloquial';
        const fallbackPrompt = this.activePersona === 'indian_priya' ? 'NATF0.pt' : 'NATM0.pt';
        const fallbackChar = this.activePersona === 'indian_priya' ? 'Warm' : 'Professional';
        const promptArea = document.getElementById('cfg-system-prompt');
        const customPrompt = promptArea ? promptArea.value.trim() : '';
        const url = `${proto}//${window.location.host}/v1/realtime?persona_id=${encodeURIComponent(this.activePersona)}&neural_voice=${encodeURIComponent(this.activeNeuralVoice || fallbackVoice)}&voice_prompt=${encodeURIComponent(this.activeVoice || fallbackPrompt)}&accent=Indian%20English&character=${encodeURIComponent(this.activeCharacter || fallbackChar)}&call_flow=${encodeURIComponent(this.activeCallFlow || 'conversational_companion')}&text_prompt=${encodeURIComponent(customPrompt)}`;
        this.socket.connect(url);
      }

      toggleMute() {
        const isMuted = this.audio.toggleMute();
        const btn = document.getElementById('btn-mic-toggle');
        btn.classList.toggle('active', isMuted);
      }

      async sendTextMessage() {
        const input = document.getElementById('user-text-input');
        const text = input.value.trim();
        if (!text) return;
        input.value = '';

        // Resume or initialize audio context on user gesture
        if (!this.audio.audioCtx) {
          try { await this.audio.initialize(); } catch (e) {}
        } else if (this.audio.audioCtx.state === 'suspended') {
          try { await this.audio.audioCtx.resume(); } catch (e) {}
        }

        if (!this.isConnected) {
          await this.startCall();
        }

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
          const currentText = this.currentAgentBubble.innerText;
          if (currentText && !currentText.endsWith(' ') && !token.startsWith(' ') && !/^[.,!?;:'")\]]/.test(token)) {
            this.currentAgentBubble.innerText += ' ' + token;
          } else {
            this.currentAgentBubble.innerText += token;
          }
          transcript.scrollTop = transcript.scrollHeight;
        }
      }

      startMetricsPolling() {
        setInterval(async () => {
          try {
            const res = await fetch('/metrics');
            if (res.ok) {
              const data = await res.json();
              const elStt = document.getElementById('telemetry-stt-ms');
              const elLlm = document.getElementById('telemetry-llm-ttft');
              const elTts = document.getElementById('telemetry-tts-ttfa');
              const elTotal = document.getElementById('telemetry-total-ttfa');
              const elBarge = document.getElementById('telemetry-barge-in');
              const elWorker = document.getElementById('telemetry-worker');

              if (elStt && data.stt_ms !== undefined) elStt.innerText = `${data.stt_ms} ms`;
              if (elLlm && data.llm_ttft_ms !== undefined) elLlm.innerText = `${data.llm_ttft_ms} ms`;
              if (elTts && data.tts_ttfa_ms !== undefined) elTts.innerText = `${data.tts_ttfa_ms} ms`;
              if (elTotal && data.total_ttfa_ms !== undefined) elTotal.innerText = `${data.total_ttfa_ms} ms`;
              if (elBarge && data.total_barge_in_events !== undefined) elBarge.innerText = data.total_barge_in_events;
              if (elWorker && data.worker_type) elWorker.innerText = data.worker_type;

              if (data.gpu) {
                const g = data.gpu;
                const devBadge = document.getElementById('gpu-device-badge');
                if (devBadge && g.device_name) devBadge.innerText = g.device_name.replace('NVIDIA GeForce ', '');
                const vramEl = document.getElementById('gpu-vram-text');
                if (vramEl && g.vram_used_mb !== undefined) {
                  vramEl.innerText = `${Math.round(g.vram_used_mb)} / ${Math.round(g.vram_total_mb)} MB`;
                }
                const tuEl = document.getElementById('gpu-temp-util');
                if (tuEl && g.temperature_c !== undefined) {
                  tuEl.innerText = `${g.temperature_c || '--'}°C / ${g.gpu_util_pct || 0}%`;
                }
              }

              // Update live stage indicators
              const stHeard = document.getElementById('stage-heard-speech');
              if (stHeard) stHeard.innerText = this.audio && !this.audio.isMuted ? (this.framesIn > 0 ? 'Active Stream' : 'Listening') : 'Muted';
              const stLLM = document.getElementById('stage-llm-thinking');
              if (stLLM) stLLM.innerText = data.llm_ttft_ms ? `${data.llm_ttft_ms} ms TTFT` : 'Idle';
              const stTTS = document.getElementById('stage-tts-speaking');
              if (stTTS) stTTS.innerText = this.agentSpeaking ? 'Speaking (24kHz)' : 'Idle';
            }
          } catch (e) {}
        }, 1500);
      }

      async toggleAudioBypass() {
        const btn = document.getElementById('btn-toggle-bypass');
        if (!this.activeSessionId) {
          this.audioBypass = !this.audioBypass;
          btn.innerText = this.audioBypass ? 'Clean: OFF' : 'Clean: ON';
          btn.classList.toggle('active', !this.audioBypass);
          return;
        }
        try {
          const res = await fetch(`/v1/audio/toggle-bypass/${this.activeSessionId}`, { method: 'POST' });
          if (res.ok) {
            const data = await res.json();
            btn.innerText = data.bypass ? 'Clean: OFF' : 'Clean: ON';
            btn.classList.toggle('active', !data.bypass);
          }
        } catch (e) {}
      }

      downloadRawWav() {
        if (this.activeSessionId) {
          window.open(`/v1/audio/raw/${this.activeSessionId}`, '_blank');
        } else {
          alert('Start an active call first to capture audio.');
        }
      }

      downloadCleanWav() {
        if (this.activeSessionId) {
          window.open(`/v1/audio/clean/${this.activeSessionId}`, '_blank');
        } else {
          alert('Start an active call first to capture audio.');
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
