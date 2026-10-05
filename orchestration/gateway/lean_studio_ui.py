"""
Lean S2S Voice Agent Studio - Single Page React 18 Application.
Provides full voice agent configuration (18 PersonaPlex voices, prompt compiler,
live token counter, voice linter, immutable versioning) and live browser 16 kHz voice call.
"""

LEAN_STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PersonaPlex Voice Agent Studio • Lean S2S</title>

  <!-- Modern Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

  <!-- React 18 + Babel Standalone -->
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

  <style>
    :root {
      --bg-base: #090a0f;
      --bg-surface: #12141c;
      --bg-surface-elevated: #1a1d29;
      --bg-surface-hover: #222638;
      --border-subtle: rgba(255, 255, 255, 0.07);
      --border-focus: rgba(99, 102, 241, 0.5);

      --accent-indigo: #6366f1;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;

      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-full: 9999px;

      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-base);
      color: var(--text-primary);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border-subtle); border-radius: var(--radius-full); }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* Layout */
    .app-header {
      height: 64px;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(18, 20, 28, 0.75);
      backdrop-filter: blur(16px);
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, var(--accent-indigo), var(--accent-cyan));
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .brand-title {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .brand-badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: var(--radius-full);
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
      font-weight: 500;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-radius: var(--radius-md);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-secondary {
      background: var(--bg-surface-elevated);
      color: var(--text-primary);
      border-color: var(--border-subtle);
    }
    .btn-secondary:hover:not(:disabled) {
      background: var(--bg-surface-hover);
      border-color: rgba(255, 255, 255, 0.15);
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-indigo), #4f46e5);
      color: white;
      box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
    }
    .btn-primary:hover:not(:disabled) {
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5);
      filter: brightness(1.1);
    }
    .btn-success {
      background: linear-gradient(135deg, var(--accent-emerald), #059669);
      color: white;
    }
    .btn-talk {
      background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
      color: white;
      padding: 9px 20px;
      font-size: 14px;
      box-shadow: 0 0 20px rgba(16, 185, 129, 0.35);
      animation: pulse-glow 3s infinite;
    }
    @keyframes pulse-glow {
      0%, 100% { box-shadow: 0 0 16px rgba(16, 185, 129, 0.3); }
      50% { box-shadow: 0 0 26px rgba(6, 182, 212, 0.55); }
    }

    .main-container {
      display: flex;
      flex: 1;
      height: calc(100vh - 64px);
    }

    /* Sidebar */
    .sidebar {
      width: 280px;
      border-right: 1px solid var(--border-subtle);
      background: var(--bg-surface);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }
    .sidebar-header {
      padding: 16px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .sidebar-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    .agent-item {
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: background 0.15s ease;
    }
    .agent-item:hover {
      background: var(--bg-surface-elevated);
    }
    .agent-item.active {
      background: rgba(99, 102, 241, 0.12);
      border-left: 3px solid var(--accent-indigo);
    }
    .agent-item-name {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .agent-item-meta {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: var(--text-secondary);
    }
    .badge {
      display: inline-block;
      padding: 1px 6px;
      border-radius: var(--radius-full);
      font-size: 10px;
      font-weight: 600;
    }
    .badge-published {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-draft {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Editor Canvas */
    .editor-canvas {
      flex: 1;
      overflow-y: auto;
      padding: 32px 48px;
      max-width: 1080px;
      margin: 0 auto;
      width: 100%;
    }

    .form-section {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin-bottom: 24px;
    }
    .section-title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-desc {
      font-size: 13px;
      color: var(--text-secondary);
      margin-bottom: 20px;
    }

    .form-group {
      margin-bottom: 20px;
    }
    .form-label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }

    input[type="text"], select, textarea {
      width: 100%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      color: var(--text-primary);
      padding: 10px 14px;
      font-size: 14px;
      font-family: inherit;
      transition: all 0.15s ease;
      outline: none;
    }
    input[type="text"]:focus, select:focus, textarea:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    textarea {
      font-family: var(--font-mono);
      line-height: 1.5;
      resize: vertical;
    }

    /* Voice Grid */
    .voice-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .voice-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 6px;
      position: relative;
      transition: all 0.15s ease;
    }
    .voice-card:hover {
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }
    .voice-card.selected {
      background: rgba(99, 102, 241, 0.12);
      border-color: var(--accent-indigo);
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.25);
    }
    .voice-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .voice-id {
      font-weight: 700;
      font-size: 13px;
    }
    .voice-tag {
      font-size: 10px;
      background: rgba(255, 255, 255, 0.08);
      padding: 2px 6px;
      border-radius: var(--radius-full);
      color: var(--text-secondary);
    }
    .voice-desc {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
    }
    .voice-preview-btn {
      background: rgba(255, 255, 255, 0.08);
      border: none;
      color: var(--text-primary);
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .voice-preview-btn:hover {
      background: var(--accent-indigo);
    }

    /* Token Meter & Linter */
    .token-bar-container {
      margin-top: 10px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 12px 14px;
    }
    .token-meter-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .token-count-text {
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 600;
    }
    .token-progress-bg {
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: var(--radius-full);
      overflow: hidden;
    }
    .token-progress-fill {
      height: 100%;
      border-radius: var(--radius-full);
      transition: width 0.3s ease, background 0.3s ease;
    }
    .token-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .token-chip {
      font-family: var(--font-mono);
      font-size: 11px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-subtle);
      color: #cbd5e1;
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .token-chip:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent-indigo);
      color: white;
    }

    .warning-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: var(--radius-md);
      font-size: 12px;
      margin-top: 8px;
    }
    .warning-pill-warn {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.25);
      color: #fde68a;
    }
    .warning-pill-error {
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.25);
      color: #fecdd3;
    }

    /* Voice Call Modal / Overlay */
    .call-overlay {
      position: fixed;
      inset: 0;
      background: rgba(9, 10, 15, 0.88);
      backdrop-filter: blur(24px);
      z-index: 100;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .call-card {
      width: 100%;
      max-width: 640px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .call-header {
      padding: 20px 24px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .call-body {
      padding: 32px 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
    }

    /* Audio Visualizer Ring / Orb */
    .visualizer-orb {
      width: 140px;
      height: 140px;
      border-radius: 50%;
      background: radial-gradient(circle at 35% 35%, #818cf8, #4f46e5 60%, #06b6d4);
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      box-shadow: 0 0 40px rgba(99, 102, 241, 0.5);
      transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .visualizer-orb.speaking {
      animation: orb-speaking 1.2s infinite ease-in-out alternate;
    }
    .visualizer-orb.listening {
      animation: orb-listening 2s infinite ease-in-out;
    }
    @keyframes orb-speaking {
      0% { transform: scale(1.0); box-shadow: 0 0 40px rgba(6, 182, 212, 0.5); }
      100% { transform: scale(1.18); box-shadow: 0 0 70px rgba(99, 102, 241, 0.85); }
    }
    @keyframes orb-listening {
      0%, 100% { transform: scale(1.0); }
      50% { transform: scale(1.06); box-shadow: 0 0 50px rgba(16, 185, 129, 0.5); }
    }

    .transcript-box {
      width: 100%;
      height: 220px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      font-size: 14px;
    }
    .turn-bubble {
      max-width: 85%;
      padding: 10px 14px;
      border-radius: var(--radius-md);
      line-height: 1.45;
    }
    .turn-assistant {
      align-self: flex-start;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #f1f5f9;
    }
    .turn-user {
      align-self: flex-end;
      background: rgba(255, 255, 255, 0.1);
      color: white;
    }

    .call-controls {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 12px;
    }
    .btn-hangup {
      background: var(--accent-rose);
      color: white;
      padding: 10px 24px;
      border-radius: var(--radius-full);
      font-weight: 700;
    }
    .btn-hangup:hover {
      background: #e11d48;
      box-shadow: 0 0 20px rgba(244, 63, 94, 0.5);
    }
  </style>
</head>
<body>

  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useRef } = React;

    // 18 Official PersonaPlex Voice Presets
    const OFFICIAL_VOICES = [
      { id: "NATF0.pt", name: "NATF0", gender: "Female", cat: "Natural", tag: "Warm & Calm", desc: "Soft-spoken, comforting teacher tone" },
      { id: "NATF1.pt", name: "NATF1", gender: "Female", cat: "Natural", tag: "Crisp & Professional", desc: "Clear diction, customer support, clinic" },
      { id: "NATF2.pt", name: "NATF2", gender: "Female", cat: "Natural", tag: "Expressive & Friendly", desc: "Standard conversational assistant" },
      { id: "NATF3.pt", name: "NATF3", gender: "Female", cat: "Natural", tag: "Bright & Articulate", desc: "Academic explanations and guidance" },
      { id: "NATM0.pt", name: "NATM0", gender: "Male", cat: "Natural", tag: "Deep & Authoritative", desc: "Corporate advisor, confident leader" },
      { id: "NATM1.pt", name: "NATM1", gender: "Male", cat: "Natural", tag: "Warm & Narrative", desc: "Conversational host, podcast guide" },
      { id: "NATM2.pt", name: "NATM2", gender: "Male", cat: "Natural", tag: "Technical & Energetic", desc: "Sales engineering, technician" },
      { id: "NATM3.pt", name: "NATM3", gender: "Male", cat: "Natural", tag: "Casual & Direct", desc: "Informal, quick responses" },
      { id: "VARF0.pt", name: "VARF0", gender: "Female", cat: "Variety", tag: "Dynamic Storyteller", desc: "High expressive range, animated" },
      { id: "VARF1.pt", name: "VARF1", gender: "Female", cat: "Variety", tag: "Upbeat Presenter", desc: "Cheerful and motivating" },
      { id: "VARF2.pt", name: "VARF2", gender: "Female", cat: "Variety", tag: "Thoughtful Analyst", desc: "Deliberate and reflective cadence" },
      { id: "VARF3.pt", name: "VARF3", gender: "Female", cat: "Variety", tag: "Melodic & Serene", desc: "Mindfulness and meditation guide" },
      { id: "VARF4.pt", name: "VARF4", gender: "Female", cat: "Variety", tag: "Vibrant Actor", desc: "Distinctive character inflections" },
      { id: "VARM0.pt", name: "VARM0", gender: "Male", cat: "Variety", tag: "Radio Announcer", desc: "Broadcaster resonance and punch" },
      { id: "VARM1.pt", name: "VARM1", gender: "Male", cat: "Variety", tag: "Space & Urgency", desc: "Command center, emergency coordinator" },
      { id: "VARM2.pt", name: "VARM2", gender: "Male", cat: "Variety", tag: "Empathetic Listener", desc: "Gentle counselor, patient guide" },
      { id: "VARM3.pt", name: "VARM3", gender: "Male", cat: "Variety", tag: "Fast Tech Host", desc: "Fast-paced conversational energy" },
      { id: "VARM4.pt", name: "VARM4", gender: "Male", cat: "Variety", tag: "Smooth Baritone", desc: "Late-night radio, deep warmth" },
    ];

    const POPULAR_TIMEZONES = [
      "Asia/Kolkata", "America/New_York", "America/Chicago",
      "America/Denver", "America/Los_Angeles", "Europe/London",
      "Europe/Paris", "Asia/Dubai", "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney", "UTC"
    ];

    function App() {
      const [agents, setAgents] = useState([]);
      const [selectedAgent, setSelectedAgent] = useState(null);
      const [loading, setLoading] = useState(true);
      const [saving, setSaving] = useState(false);
      const [publishing, setPublishing] = useState(false);
      const [callActive, setCallActive] = useState(false);

      // Live compiled token metrics & warnings
      const [compiledMetrics, setCompiledMetrics] = useState({
        tokenCount: 0,
        canPublish: true,
        warnings: [],
        timeLine: "",
        dayPart: "morning"
      });

      // Load agents on boot
      useEffect(() => {
        loadAgents();
      }, []);

      const loadAgents = async () => {
        try {
          const res = await fetch("/v1/agents");
          const data = await res.json();
          const items = data.items || data.agents || [];
          setAgents(items);
          if (items.length > 0 && !selectedAgent) {
            loadAgentDetails(items[0].id);
          } else if (items.length === 0) {
            createNewAgent();
          }
        } catch (e) {
          console.error("Failed to load agents", e);
        } finally {
          setLoading(false);
        }
      };

      const loadAgentDetails = async (id) => {
        try {
          const res = await fetch(`/v1/agents/${id}`);
          if (res.ok) {
            const data = await res.json();
            setSelectedAgent(data);
            triggerCompile(data);
          }
        } catch (e) {
          console.error("Failed to load agent details", e);
        }
      };

      const triggerCompile = async (agent) => {
        if (!agent) return;
        try {
          const res = await fetch("/v1/prompts/compile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              system_prompt: agent.draft_system_prompt || "",
              greeting_text: agent.draft_greeting_text || "",
              greeting_mode: agent.draft_greeting_mode || "agent_first",
              ending_text: agent.draft_ending_text || "",
              agent_name: agent.name || "Assistant",
              timezone: agent.draft_timezone || "Asia/Kolkata",
            })
          });
          if (res.ok) {
            const data = await res.json();
            setCompiledMetrics({
              tokenCount: data.token_count,
              canPublish: data.can_publish,
              warnings: data.warnings || [],
              timeLine: data.time_line,
              dayPart: data.day_part,
            });
          }
        } catch (e) {
          console.warn("Compile preview failed", e);
        }
      };

      const createNewAgent = async () => {
        try {
          const res = await fetch("/v1/agents", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: "New Voice Agent",
              voice_id: "NATF2.pt",
              greeting_mode: "agent_first",
              greeting_text: "Hello! How can I help you today?",
              system_prompt: "You are a warm, helpful assistant. Answer questions concisely.",
              ending_text: "Thank you for speaking with me. Goodbye!",
              timezone: "Asia/Kolkata",
              auto_publish: false,
            })
          });
          if (res.ok) {
            const created = await res.json();
            await loadAgents();
            setSelectedAgent(created);
            triggerCompile(created);
          }
        } catch (e) {
          alert("Error creating agent: " + e.message);
        }
      };

      const saveDraft = async () => {
        if (!selectedAgent) return;
        setSaving(true);
        try {
          const res = await fetch(`/v1/agents/${selectedAgent.id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: selectedAgent.name,
              voice_id: selectedAgent.draft_voice_id,
              greeting_mode: selectedAgent.draft_greeting_mode,
              greeting_text: selectedAgent.draft_greeting_text,
              system_prompt: selectedAgent.draft_system_prompt,
              ending_text: selectedAgent.draft_ending_text,
              end_silence_sec: selectedAgent.draft_end_silence_sec,
              max_duration_sec: selectedAgent.draft_max_duration_sec,
              timezone: selectedAgent.draft_timezone,
            })
          });
          if (res.ok) {
            const updated = await res.json();
            setSelectedAgent(updated.agent || updated);
            triggerCompile(updated.agent || updated);
            loadAgents();
          }
        } finally {
          setSaving(false);
        }
      };

      const publishVersion = async () => {
        if (!selectedAgent || !compiledMetrics.canPublish) return;
        setPublishing(true);
        try {
          const res = await fetch(`/v1/agents/${selectedAgent.id}/publish`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              change_note: `Published via Studio (${compiledMetrics.tokenCount} tokens)`
            })
          });
          if (res.ok) {
            await loadAgentDetails(selectedAgent.id);
            await loadAgents();
          } else {
            const err = await res.json();
            alert("Publish blocked: " + (err.detail || "Exceeds token limit"));
          }
        } finally {
          setPublishing(false);
        }
      };

      const updateField = (field, val) => {
        setSelectedAgent(prev => {
          const next = { ...prev, [field]: val };
          triggerCompile(next);
          return next;
        });
      };

      const playVoicePreview = (voiceId, e) => {
        e.stopPropagation();
        const audio = new Audio(`/v1/voices/${voiceId}/preview`);
        audio.play().catch(e => console.log("Preview audio failed:", e));
      };

      return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
          {/* Top Bar */}
          <header className="app-header">
            <div className="brand">
              <div className="brand-logo">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
              </div>
              <div>
                <span className="brand-title">PersonaPlex Studio</span>
                <span className="brand-badge" style={{ marginLeft: 8 }}>Lean S2S • 16 kHz</span>
              </div>
            </div>

            <div className="header-actions">
              <button className="btn btn-secondary" onClick={saveDraft} disabled={saving}>
                {saving ? "Saving..." : "Save Draft"}
              </button>
              <button
                className="btn btn-primary"
                onClick={publishVersion}
                disabled={publishing || !compiledMetrics.canPublish}
                title={!compiledMetrics.canPublish ? "Exceeds 350 tokens" : "Create immutable version"}
              >
                {publishing ? "Publishing..." : `Publish v${(selectedAgent?.current_version_no || 0) + 1}`}
              </button>
              <button className="btn btn-talk" onClick={() => setCallActive(true)} disabled={!selectedAgent}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                </svg>
                Talk to Agent
              </button>
            </div>
          </header>

          {/* Main Workspace */}
          <div className="main-container">
            {/* Sidebar Agent List */}
            <aside className="sidebar">
              <div className="sidebar-header">
                <span className="sidebar-title">Your Agents</span>
                <button className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: 11 }} onClick={createNewAgent}>
                  + New
                </button>
              </div>
              <div style={{ flex: 1 }}>
                {agents.map(a => (
                  <div
                    key={a.id}
                    className={`agent-item ${selectedAgent?.id === a.id ? 'active' : ''}`}
                    onClick={() => loadAgentDetails(a.id)}
                  >
                    <div className="agent-item-name">{a.name}</div>
                    <div className="agent-item-meta">
                      <span className={`badge ${a.status === 'published' ? 'badge-published' : 'badge-draft'}`}>
                        {a.status === 'published' ? `v${a.current_version_no}` : 'Draft'}
                      </span>
                      <span>•</span>
                      <span>{a.draft_voice_id || a.voice_prompt}</span>
                    </div>
                  </div>
                ))}
              </div>
            </aside>

            {/* Main Configuration Canvas */}
            <main className="editor-canvas">
              {selectedAgent ? (
                <div>
                  {/* General Card */}
                  <div className="form-section">
                    <div className="section-title">General & Voice Persona</div>
                    <div className="section-desc">Select the PersonaPlex voice preset and configure agent identity.</div>

                    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
                      <div className="form-group">
                        <label className="form-label">Agent Name</label>
                        <input
                          type="text"
                          value={selectedAgent.name || ""}
                          onChange={e => updateField("name", e.target.value)}
                          placeholder="e.g. Clinic Receptionist"
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Timezone (IANA)</label>
                        <select
                          value={selectedAgent.draft_timezone || "Asia/Kolkata"}
                          onChange={e => updateField("draft_timezone", e.target.value)}
                        >
                          {POPULAR_TIMEZONES.map(tz => (
                            <option key={tz} value={tz}>{tz}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Select Voice (18 Official PersonaPlex Presets)</label>
                      <div className="voice-grid">
                        {OFFICIAL_VOICES.map(v => (
                          <div
                            key={v.id}
                            className={`voice-card ${selectedAgent.draft_voice_id === v.id ? 'selected' : ''}`}
                            onClick={() => updateField("draft_voice_id", v.id)}
                          >
                            <div className="voice-card-top">
                              <span className="voice-id">{v.name} ({v.gender[0]})</span>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span className="voice-tag">{v.cat}</span>
                                <button
                                  className="voice-preview-btn"
                                  onClick={(e) => playVoicePreview(v.id, e)}
                                  title="Listen to preview"
                                >
                                  ▶
                                </button>
                              </div>
                            </div>
                            <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 500 }}>{v.tag}</div>
                            <div className="voice-desc">{v.desc}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Greeting Card */}
                  <div className="form-section">
                    <div className="section-title">Greeting & Turn-Taking</div>
                    <div className="section-desc">Decide who speaks first when the audio session starts.</div>

                    <div style={{ display: 'flex', gap: 20, marginBottom: 16 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13 }}>
                        <input
                          type="radio"
                          name="greeting_mode"
                          checked={selectedAgent.draft_greeting_mode !== "user_first"}
                          onChange={() => updateField("draft_greeting_mode", "agent_first")}
                        />
                        <span>Agent speaks first (Default)</span>
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13 }}>
                        <input
                          type="radio"
                          name="greeting_mode"
                          checked={selectedAgent.draft_greeting_mode === "user_first"}
                          onChange={() => updateField("draft_greeting_mode", "user_first")}
                        />
                        <span>User speaks first (Wait for caller)</span>
                      </label>
                    </div>

                    {selectedAgent.draft_greeting_mode !== "user_first" && (
                      <div className="form-group">
                        <label className="form-label">Greeting Line</label>
                        <input
                          type="text"
                          value={selectedAgent.draft_greeting_text || ""}
                          onChange={e => updateField("draft_greeting_text", e.target.value)}
                          placeholder="e.g. Hello, thank you for calling Metro Health. How can I help you today?"
                        />
                      </div>
                    )}
                  </div>

                  {/* System Prompt & Live Token Meter */}
                  <div className="form-section">
                    <div className="section-title">System Prompt & Live Voice Linter</div>
                    <div className="section-desc">
                      Condition the PersonaPlex S2S model. Keep prompts concise (ideal ≤150 tokens, hard limit 350 tokens).
                    </div>

                    <div className="form-group">
                      <textarea
                        rows={6}
                        value={selectedAgent.draft_system_prompt || ""}
                        onChange={e => updateField("draft_system_prompt", e.target.value)}
                        placeholder="You are a compassionate receptionist at Metro Clinic. Help the caller schedule appointments..."
                      />

                      {/* Token Progress Bar & Indicator */}
                      <div className="token-bar-container">
                        <div className="token-meter-top">
                          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                            SentencePiece Token Budget (Official 32k Vocab):
                          </span>
                          <span
                            className="token-count-text"
                            style={{
                              color: compiledMetrics.tokenCount > 350 ? 'var(--accent-rose)' :
                                     compiledMetrics.tokenCount > 150 ? 'var(--accent-amber)' : 'var(--accent-emerald)'
                            }}
                          >
                            {compiledMetrics.tokenCount} / 350 tokens {compiledMetrics.tokenCount <= 150 ? '(Ideal)' : ''}
                          </span>
                        </div>
                        <div className="token-progress-bg">
                          <div
                            className="token-progress-fill"
                            style={{
                              width: `${Math.min(100, (compiledMetrics.tokenCount / 350) * 100)}%`,
                              background: compiledMetrics.tokenCount > 350 ? 'var(--accent-rose)' :
                                          compiledMetrics.tokenCount > 150 ? 'var(--accent-amber)' : 'var(--accent-emerald)'
                            }}
                          />
                        </div>

                        {/* Variable insertion chips */}
                        <div className="token-chips">
                          <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>Insert variable:</span>
                          {["{{agent_name}}", "{{current_time}}", "{{weekday}}", "{{day_part}}", "{{caller_name}}"].map(tag => (
                            <span
                              key={tag}
                              className="token-chip"
                              onClick={() => {
                                const cur = selectedAgent.draft_system_prompt || "";
                                updateField("draft_system_prompt", cur + " " + tag);
                              }}
                            >
                              + {tag}
                            </span>
                          ))}
                        </div>

                        {/* Live Linter Warnings */}
                        {compiledMetrics.warnings.map((w, i) => (
                          <div key={i} className={`warning-pill ${w.includes('exceeds hard limit') ? 'warning-pill-error' : 'warning-pill-warn'}`}>
                            <span>⚠️</span>
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Call Ending & Timeouts */}
                  <div className="form-section">
                    <div className="section-title">Call Ending & Natural Closure</div>
                    <div className="section-desc">Automatic hangup detection via fuzzy text matching and timeouts.</div>

                    <div className="form-group">
                      <label className="form-label">Goodbye Line (Natural Hangup Trigger)</label>
                      <input
                        type="text"
                        value={selectedAgent.draft_ending_text || ""}
                        onChange={e => updateField("draft_ending_text", e.target.value)}
                        placeholder="e.g. Thank you for calling. Have a wonderful day, goodbye!"
                      />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                      <div className="form-group">
                        <label className="form-label">
                          Silence Timeout: {selectedAgent.draft_end_silence_sec || 20}s
                        </label>
                        <input
                          type="range"
                          min="5"
                          max="60"
                          value={selectedAgent.draft_end_silence_sec || 20}
                          onChange={e => updateField("draft_end_silence_sec", parseInt(e.target.value))}
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">
                          Max Call Duration: {Math.round((selectedAgent.draft_max_duration_sec || 600) / 60)} min
                        </label>
                        <input
                          type="range"
                          min="60"
                          max="1800"
                          step="60"
                          value={selectedAgent.draft_max_duration_sec || 600}
                          onChange={e => updateField("draft_max_duration_sec", parseInt(e.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', marginTop: 100, color: 'var(--text-muted)' }}>
                  Loading agent configuration...
                </div>
              )}
            </main>
          </div>

          {/* Voice Call Modal */}
          {callActive && (
            <VoiceCallModal
              agent={selectedAgent}
              onClose={() => setCallActive(false)}
            />
          )}
        </div>
      );
    }

    // Voice Call Session Component
    function VoiceCallModal({ agent, onClose }) {
      const [status, setStatus] = useState("Connecting...");
      const [transcripts, setTranscripts] = useState([]);
      const [callDuration, setCallDuration] = useState(0);
      const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
      const [isUserSpeaking, setIsUserSpeaking] = useState(false);
      const [callEndedInfo, setCallEndedInfo] = useState(null);

      const wsRef = useRef(null);
      const audioCtxRef = useRef(null);
      const micStreamRef = useRef(null);
      const transcriptBoxRef = useRef(null);
      const timerRef = useRef(null);

      useEffect(() => {
        startCallSession();
        return () => endCallSession();
      }, []);

      useEffect(() => {
        if (transcriptBoxRef.current) {
          transcriptBoxRef.current.scrollTop = transcriptBoxRef.current.scrollHeight;
        }
      }, [transcripts]);

      const startCallSession = async () => {
        try {
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          const audioCtx = new AudioContext({ sampleRate: 16000 });
          audioCtxRef.current = audioCtx;

          // Microphone 16kHz
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              sampleRate: 16000,
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            }
          });
          micStreamRef.current = stream;

          // Connect WebSocket to /v2/voice
          const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
          const wsUrl = `${protocol}//${window.location.host}/v2/voice?agent_id=${agent.id}&sample_rate=16000&codec=pcm16`;
          const ws = new WebSocket(wsUrl);
          wsRef.current = ws;
          ws.binaryType = "arraybuffer";

          ws.onopen = () => {
            setStatus("Connected • Listening");
            // Start timer
            const t0 = Date.now();
            timerRef.current = setInterval(() => {
              setCallDuration(Math.floor((Date.now() - t0) / 1000));
            }, 1000);

            // Hook mic capture to ws
            setupMicAudioCapture(stream, audioCtx, ws);
          };

          ws.onmessage = (event) => {
            if (typeof event.data === "string") {
              try {
                const msg = JSON.parse(event.data);
                if (msg.type === "transcript") {
                  setIsAgentSpeaking(true);
                  setTranscripts(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.role === "assistant" && !last.final) {
                      return [...prev.slice(0, -1), { ...last, text: last.text + msg.text }];
                    } else {
                      return [...prev, { role: "assistant", text: msg.text }];
                    }
                  });
                  setTimeout(() => setIsAgentSpeaking(false), 800);
                } else if (msg.type === "call_ended") {
                  setCallEndedInfo(msg);
                  setStatus(`Call Ended (${msg.reason})`);
                }
              } catch (e) {}
            } else if (event.data instanceof ArrayBuffer) {
              // Playback 16 kHz PCM
              setIsAgentSpeaking(true);
              playPCM16(event.data, audioCtx);
              setTimeout(() => setIsAgentSpeaking(false), 500);
            }
          };

          ws.onclose = () => {
            if (!callEndedInfo) {
              setStatus("Call Finished");
            }
          };

        } catch (e) {
          setStatus("Microphone permission or connection failed: " + e.message);
        }
      };

      const setupMicAudioCapture = (stream, audioCtx, ws) => {
        const source = audioCtx.createMediaStreamSource(stream);
        // Process in 20ms frames = 320 samples at 16kHz
        const bufferSize = 512;
        const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

        let accumulator = [];
        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const inputData = e.inputBuffer.getChannelData(0);

          // Check user speaking energy
          let sumSq = 0;
          for (let i = 0; i < inputData.length; i++) sumSq += inputData[i] * inputData[i];
          const rms = Math.sqrt(sumSq / inputData.length);
          setIsUserSpeaking(rms > 0.015);

          for (let i = 0; i < inputData.length; i++) accumulator.push(inputData[i]);

          while (accumulator.length >= 320) {
            const frame = accumulator.slice(0, 320);
            accumulator = accumulator.slice(320);

            // Convert to int16 PCM
            const pcm16 = new Int16Array(320);
            for (let i = 0; i < 320; i++) {
              const s = Math.max(-1, Math.min(1, frame[i]));
              pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            ws.send(pcm16.buffer);
          }
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);
      };

      let scheduledTime = 0;
      const playPCM16 = (arrayBuffer, audioCtx) => {
        const pcm16 = new Int16Array(arrayBuffer);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
          float32[i] = pcm16[i] / 32768.0;
        }

        const buffer = audioCtx.createBuffer(1, float32.length, 16000);
        buffer.getChannelData(0).set(float32);

        const node = audioCtx.createBufferSource();
        node.buffer = buffer;
        node.connect(audioCtx.destination);

        const now = audioCtx.currentTime;
        if (scheduledTime < now) scheduledTime = now + 0.02;
        node.start(scheduledTime);
        scheduledTime += buffer.duration;
      };

      const endCallSession = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          try {
            wsRef.current.send(JSON.stringify({ type: "hangup" }));
            wsRef.current.close();
          } catch (e) {}
        }
        if (micStreamRef.current) {
          micStreamRef.current.getTracks().forEach(t => t.stop());
        }
        if (audioCtxRef.current) {
          audioCtxRef.current.close();
        }
      };

      const formatDuration = (sec) => {
        const m = Math.floor(sec / 60).toString().padStart(2, '0');
        const s = (sec % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
      };

      return (
        <div className="call-overlay">
          <div className="call-card">
            <div className="call-header">
              <div>
                <div style={{ fontWeight: 700, fontSize: 16 }}>{agent?.name || "Voice Agent"}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  16 kHz Full-Duplex S2S • {formatDuration(callDuration)}
                </div>
              </div>
              <div className="badge badge-published" style={{ padding: '4px 10px' }}>
                {status}
              </div>
            </div>

            <div className="call-body">
              {/* Dynamic Voice Orb */}
              <div className={`visualizer-orb ${isAgentSpeaking ? 'speaking' : isUserSpeaking ? 'listening' : ''}`}>
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
              </div>

              {/* Streaming Transcript Log */}
              <div className="transcript-box" ref={transcriptBoxRef}>
                {transcripts.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', textAlign: 'center', margin: 'auto' }}>
                    Speak into your microphone. Conversation turns will stream here in real-time.
                  </div>
                ) : (
                  transcripts.map((t, idx) => (
                    <div key={idx} className={`turn-bubble ${t.role === 'assistant' ? 'turn-assistant' : 'turn-user'}`}>
                      <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 2, color: t.role === 'assistant' ? '#818cf8' : '#94a3b8' }}>
                        {t.role === 'assistant' ? agent?.name : 'You'}
                      </div>
                      <div>{t.text}</div>
                    </div>
                  ))
                )}
              </div>

              {/* End Reason Notice */}
              {callEndedInfo && (
                <div style={{ padding: '8px 16px', background: 'rgba(255,255,255,0.06)', borderRadius: 8, fontSize: 13, color: '#38bdf8' }}>
                  Call ended: <strong>{callEndedInfo.reason}</strong>. {callEndedInfo.message}
                </div>
              )}

              {/* Call Controls */}
              <div className="call-controls">
                <button className="btn btn-hangup" onClick={() => { endCallSession(); onClose(); }}>
                  End Call
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<App />);
  </script>
</body>
</html>
"""
