/* =====================================================================
   Arvis - Web frontend main controller v2
   Wires Socket.IO for live updates, REST for chat, Web Speech API for
   always-on wake-word input + browser-side TTS.
   ===================================================================== */

(function () {
    "use strict";

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------
    const $ = (id) => document.getElementById(id);
    function qs(sel, root) { return (root || document).querySelector(sel); }
    function qsa(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

    function formatTime(ts) {
        return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function glitch(el) {
        if (!el) return;
        el.classList.remove("glitch");
        void el.offsetWidth;
        el.classList.add("glitch");
    }

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------
    const state = {
        listening: false,
        thinking: false,
        speaking: false,
        history: [],
        recent: [],
        tools: [],
        modes: [
            { id: "default", name: "DEFAULT", emoji: "◆", colour: "#5fe4ff", prompt: "" },
            { id: "focus",   name: "FOCUS",   emoji: "◉", colour: "#6fffa2", prompt: "Be concise and focused." },
            { id: "study",   name: "STUDY",   emoji: "✎", colour: "#ffd166", prompt: "Explain clearly, step by step." },
            { id: "coder",   name: "CODER",   emoji: "⌬", colour: "#ff6acb", prompt: "Reply like a senior engineer." },
            { id: "netrun",  name: "NETRUN",  emoji: "⌖", colour: "#b388ff", prompt: "Be terse and technical." },
        ],
        activeMode: "default",
        routines: [
            { id: "morning",  label: "MORNING BRIEF",  cmd: "run morning routine" },
            { id: "shutdown", label: "SHUTDOWN",       cmd: "run shutdown routine" },
            { id: "focus",    label: "FOCUS MODE",     cmd: "activate focus mode" },
            { id: "scan",     label: "SCAN NETWORK",   cmd: "arp scan" },
            { id: "shot",     label: "SCREENSHOT",     cmd: "take a screenshot" },
        ],
        customCommands: [],
        customModes: [],
        reminders: [],
        google: { configured: false, signed_in: false, email: null },
        triggerWord: "arvis",
        conversationMode: false,
        conversationUntil: 0,
        // TTS
        ttsEnabled: true,
        ttsRate: 1.0,
        ttsPitch: 1.0,
        ttsVoice: null,
        ttsVolume: 1.0,
    };

    // -----------------------------------------------------------------
    // Audio analyser - feeds the orb
    // -----------------------------------------------------------------
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let speakerStream = null;
    const audioState = { energy: 0, bass: 0, wave: null };
    window.__arvisAudio = audioState;

    function ensureAudioCtx() {
        if (audioCtx) return audioCtx;
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn("AudioContext unavailable:", e);
            return null;
        }
        return audioCtx;
    }

    async function attachMicAnalyser(stream) {
        const ctx = ensureAudioCtx();
        if (!ctx) return;
        if (analyser) try { analyser.disconnect(); } catch {}
        const src = ctx.createMediaStreamSource(stream);
        analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.7;
        src.connect(analyser);
        window.__arvisAnalyser = analyser;
    }

    function readAudio() {
        if (!analyser) return;
        const buf = new Uint8Array(analyser.fftSize);
        analyser.getByteFrequencyData(buf);
        let sum = 0, bassSum = 0;
        const bassEnd = Math.floor(buf.length * 0.25);
        for (let i = 0; i < buf.length; i++) {
            const v = buf[i] / 255;
            sum += v;
            if (i < bassEnd) bassSum += v;
        }
        audioState.energy = sum / buf.length;
        audioState.bass = bassSum / bassEnd;

        // Time-domain waveform for ring (downsample to 64 points)
        const td = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(td);
        const wave = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
            const start = Math.floor(i * td.length / 64);
            const end = Math.floor((i + 1) * td.length / 64);
            let max = 0;
            for (let k = start; k < end; k++) {
                const v = Math.abs(td[k] - 128) / 128;
                if (v > max) max = v;
            }
            wave[i] = max;
        }
        audioState.wave = wave;
    }

    function audioLoop() {
        readAudio();
        requestAnimationFrame(audioLoop);
    }
    audioLoop();

    // -----------------------------------------------------------------
    // Socket.IO
    // -----------------------------------------------------------------
    let socket = null;

    function connectSocket() {
        if (typeof io !== "function") return;
        socket = io({ transports: ["websocket", "polling"] });

        socket.on("connect", () => {
            logEvent("ok", "WebSocket connected");
            const el = $("footer-status"); if (el) el.textContent = "Live. Online.";
        });

        socket.on("disconnect", () => {
            logEvent("warn", "WebSocket disconnected");
            const el = $("footer-status"); if (el) el.textContent = "Reconnecting...";
        });

        socket.on("chat", (msg) => {
            if (!msg) {
                renderChat();
                return;
            }
            const existing = state.history.findIndex((m) => m.ts === msg.ts && m.role === msg.role);
            if (existing === -1) {
                state.history.push(msg);
                appendChatMessage(msg);
                if (msg.role === "user") {
                    state.recent.unshift({ ts: msg.ts, text: msg.content });
                    if (state.recent.length > 12) state.recent.length = 12;
                    renderRecent();
                } else if (msg.role === "assistant") {
                    speak(msg.content);
                }
            }
        });

        socket.on("state", (s) => applyRuntimeState(s));
        socket.on("system", (s) => applySystem(s));

        socket.on("custom:updated", (payload) => {
            if (!payload) return;
            if (payload.kind === "command") {
                refreshCustomCommands();
            } else if (payload.kind === "mode") {
                refreshCustomModes();
            } else if (payload.kind === "reminder") {
                refreshReminders();
            }
        });

        socket.on("reminder:fired", (rem) => {
            logEvent("warn", "⏰ Reminder fired: " + (rem.text || ""));
            speak("Reminder: " + (rem.text || ""));
        });
    }

    // -----------------------------------------------------------------
    // Chat rendering
    // -----------------------------------------------------------------
    const chatScroll = $("chat-scroll");

    function renderChat() {
        chatScroll.innerHTML = "";
        state.history.forEach(appendChatMessage);
    }

    function appendChatMessage(msg) {
        const div = document.createElement("div");
        const isReminder = msg.meta && msg.meta.reminder;
        div.className = "chat-msg " + msg.role + (isReminder ? " reminder" : "");
        const meta = msg.role === "user" ? "YOU" : (isReminder ? "REMINDER" : "ARVIS");
        div.innerHTML =
            `<span class="chat-meta">${meta} · ${formatTime(msg.ts)}</span>` +
            escapeHtml(msg.content);
        chatScroll.appendChild(div);
        const nearBottom =
            chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight < 200;
        if (nearBottom) chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    // -----------------------------------------------------------------
    // Recent
    // -----------------------------------------------------------------
    function renderRecent() {
        const ul = $("recent-list");
        ul.innerHTML = "";
        state.recent.slice(0, 12).forEach((r) => {
            const li = document.createElement("li");
            li.innerHTML = `<span class="txt">${escapeHtml(r.text)}</span><span class="ts">${formatTime(r.ts)}</span>`;
            li.addEventListener("click", () => {
                $("composer-input").value = r.text;
                $("composer-input").focus();
            });
            ul.appendChild(li);
        });
        $("recent-count").textContent = state.recent.length;
    }

    // -----------------------------------------------------------------
    // Modes / routines
    // -----------------------------------------------------------------
    function renderModes() {
        const wrap = $("modes-list");
        wrap.innerHTML = "";
        const all = [...state.modes, ...state.customModes];
        all.forEach((m) => {
            const b = document.createElement("button");
            b.className = "chip" + (m.id === state.activeMode ? " active" : "");
            b.type = "button";
            b.style.setProperty("--chip-colour", m.colour || "#5fe4ff");
            if (m.id === state.activeMode) {
                b.style.borderColor = m.colour || "#5fe4ff";
                b.style.color = m.colour || "#5fe4ff";
                b.style.boxShadow = `0 0 16px ${m.colour || "#5fe4ff"}`;
            }
            b.innerHTML = `${m.emoji || "◆"} ${escapeHtml(m.name.toUpperCase ? m.name.toUpperCase() : m.id.toUpperCase())}`;
            b.addEventListener("click", () => activateMode(m));
            wrap.appendChild(b);
        });
        $("kv-mode").textContent = (all.find((m) => m.id === state.activeMode) || {}).name || state.activeMode.toUpperCase();
    }

    function activateMode(m) {
        state.activeMode = m.id;
        const hex = (m.colour || "#5fe4ff").replace("#", "");
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        const hue = Math.round(180 + Math.atan2(g - b, r - b) * 180 / Math.PI);
        if (window.__arvisOrb) window.__arvisOrb.setHue(hue);
        glitch(qs(".orb-frame"));
        renderModes();
        // Notify the LLM via the chat dispatch.
        sendCommand(`switch mode to ${m.id}`, { silent: true });
    }

    function renderRoutines() {
        const wrap = $("routines-list");
        wrap.innerHTML = "";
        state.routines.forEach((r) => {
            const b = document.createElement("button");
            b.className = "chip";
            b.type = "button";
            b.textContent = r.label;
            b.addEventListener("click", () => sendCommand(r.cmd));
            wrap.appendChild(b);
        });
    }

    // -----------------------------------------------------------------
    // Tools
    // -----------------------------------------------------------------
    function renderTools() {
        const ul = $("tools-list");
        ul.innerHTML = "";
        state.tools.forEach((name) => {
            const li = document.createElement("li");
            li.innerHTML = `<span class="txt">${escapeHtml(name)}</span><span class="ts">◆</span>`;
            ul.appendChild(li);
        });
        $("tools-count").textContent = state.tools.length;
        $("kv-tools").textContent = state.tools.length;
    }

    // -----------------------------------------------------------------
    // System telemetry
    // -----------------------------------------------------------------
    function applySystem(s) {
        if (!s) return;
        const cpu = s.cpu_percent || 0;
        const mem = s.memory_percent || 0;
        $("meter-cpu").style.width = cpu + "%";
        $("meter-cpu-val").textContent = cpu.toFixed(0) + "%";
        $("meter-mem").style.width = mem + "%";
        $("meter-mem-val").textContent = mem.toFixed(0) + "%";

        if (s.battery) {
            $("meter-bat").style.width = s.battery.percent + "%";
            $("meter-bat-val").textContent = s.battery.percent.toFixed(0) + "%";
        } else {
            $("meter-bat").style.width = "0%";
            $("meter-bat-val").textContent = "AC";
        }

        $("kv-uptime").textContent = formatUptime(s.uptime_seconds);

        const pill = $("ollama-pill");
        if (s.ollama_reachable) {
            pill.textContent = "OLLAMA ●";
            pill.classList.add("pill-cyan");
        } else {
            pill.textContent = "OLLAMA ○";
            pill.classList.remove("pill-cyan");
            pill.style.color = "var(--text-dim)";
        }
    }

    function formatUptime(secs) {
        if (!secs && secs !== 0) return "—";
        const h = Math.floor(secs / 3600);
        const m = Math.floor((secs % 3600) / 60);
        const s = secs % 60;
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }

    // -----------------------------------------------------------------
    // TTS - browser-side speech synthesis
    // -----------------------------------------------------------------
    function pickVoice() {
        if (!window.speechSynthesis) return null;
        const voices = window.speechSynthesis.getVoices();
        if (!voices.length) return null;
        const prefs = [
            (v) => /neural|premium|enhanced/i.test(v.name) && /en[-_]?GB/i.test(v.lang),
            (v) => /neural|premium|enhanced/i.test(v.name) && /en[-_]?US/i.test(v.lang),
            (v) => /Google/.test(v.name) && /en[-_]?GB/i.test(v.lang),
            (v) => /Google/.test(v.name) && /en[-_]?US/i.test(v.lang),
            (v) => /en[-_]?GB/i.test(v.lang),
            (v) => /en[-_]?US/i.test(v.lang),
            () => true,
        ];
        for (const pred of prefs) {
            const v = voices.find(pred);
            if (v) return v;
        }
        return voices[0] || null;
    }

    function _stripForSpeech(s) {
        return String(s || "")
            .replace(/```[\s\S]*?```/g, " ")
            .replace(/`[^`]*`/g, " ")
            .replace(/\*\*([^*]+)\*\*/g, "$1")
            .replace(/\*([^*]+)\*/g, "$1")
            .replace(/_([^_]+)_/g, "$1")
            .replace(/[#>\-]+/g, " ")
            .replace(/\s+/g, " ")
            .trim();
    }

    function speak(text) {
        if (!state.ttsEnabled) return;
        if (!window.speechSynthesis) {
            logEvent("warn", "SpeechSynthesis API not available");
            return;
        }
        const cleaned = _stripForSpeech(text);
        if (!cleaned) return;
        try {
            window.speechSynthesis.cancel();
            const utter = new SpeechSynthesisUtterance(cleaned);
            utter.rate = state.ttsRate;
            utter.pitch = state.ttsPitch;
            utter.volume = state.ttsVolume;
            if (!state.ttsVoice) state.ttsVoice = pickVoice();
            if (state.ttsVoice) utter.voice = state.ttsVoice;
            utter.onstart = () => {
                state.speaking = true;
                if (window.__arvisOrb) window.__arvisOrb.setSpeaking(true);
                document.body.setAttribute("data-speaking", "on");
            };
            utter.onend = () => {
                state.speaking = false;
                if (window.__arvisOrb) window.__arvisOrb.setSpeaking(false);
                document.body.setAttribute("data-speaking", "off");
            };
            utter.onerror = () => {
                state.speaking = false;
                if (window.__arvisOrb) window.__arvisOrb.setSpeaking(false);
            };
            window.speechSynthesis.speak(utter);
        } catch (err) {
            logEvent("err", "TTS threw: " + err.message);
        }
    }

    function stopSpeaking() {
        if (window.speechSynthesis) {
            try { window.speechSynthesis.cancel(); } catch {}
        }
        state.speaking = false;
        if (window.__arvisOrb) window.__arvisOrb.setSpeaking(false);
    }

    if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = () => {
            state.ttsVoice = pickVoice();
        };
    }

    // -----------------------------------------------------------------
    // Runtime state
    // -----------------------------------------------------------------
    function applyRuntimeState(s) {
        if (!s) return;
        state.listening = !!s.listening;
        state.thinking = !!s.thinking;
        const pill = $("status-pill");
        if (s.listening && state.conversationMode) {
            pill.dataset.state = "listening";
            pill.textContent = "● LISTENING";
        } else if (s.listening) {
            pill.dataset.state = "wake";
            pill.textContent = "● WAKE";
        } else if (s.thinking) {
            pill.dataset.state = "thinking";
            pill.textContent = "● THINKING";
        } else {
            pill.dataset.state = "idle";
            pill.textContent = "● IDLE";
        }

        $("core-state").textContent = s.listening
            ? (state.conversationMode ? "LISTENING" : "WAKE")
            : s.thinking ? "PROCESSING" : "STANDBY";
        $("core-listening").textContent = s.listening ? "YES" : "NO";
        $("core-tool").textContent = s.last_tool || "—";

        if (window.__arvisOrb) {
            window.__arvisOrb.setListening(s.listening && state.conversationMode);
            window.__arvisOrb.setThinking(s.thinking);
        }

        const mic = $("mic-btn");
        if (s.listening && state.conversationMode) {
            mic.classList.add("listening");
            mic.classList.remove("wake");
        } else {
            mic.classList.remove("listening");
            mic.classList.add("wake");
        }
    }

    // -----------------------------------------------------------------
    // Event log
    // -----------------------------------------------------------------
    function logEvent(level, text) {
        const ul = $("event-log");
        if (!ul) return;
        const li = document.createElement("li");
        li.className = level;
        li.textContent = `[${new Date().toLocaleTimeString(undefined, { hour12: false })}] ${text}`;
        ul.insertBefore(li, ul.firstChild);
        while (ul.children.length > 60) ul.removeChild(ul.lastChild);
    }

    // -----------------------------------------------------------------
    // Chat send / receive
    // -----------------------------------------------------------------
    async function sendCommand(text, opts) {
        opts = opts || {};
        const cleaned = (text || "").trim();
        if (!cleaned) return;
        if (!opts.silent) logEvent("info", `→ ${cleaned}`);
        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleaned }),
            });
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            const data = await resp.json();
            if (data.history) {
                state.history = data.history;
                renderChat();
                const lastAssistant = [...state.history].reverse().find((m) => m.role === "assistant");
                if (lastAssistant && !opts.silent) speak(lastAssistant.content);
            }
            if (!opts.silent) logEvent("ok", `← ${(data.message || "").slice(0, 80)}`);
            return data;
        } catch (err) {
            logEvent("err", "send failed: " + err.message);
            return null;
        }
    }

    // -----------------------------------------------------------------
    // Always-on wake word + Web Speech API
    // -----------------------------------------------------------------
    let recognizer = null;
    let recognizing = false;
    let wakeMode = false;          // off until user grants mic permission
    let pendingArm = null;         // setInterval handle waiting for arm signal

    function enterConversationMode() {
        state.conversationMode = true;
        state.conversationUntil = Date.now() + 15000;
        glitch(qs(".orb-frame"));
        logEvent("ok", "wake word detected — listening");
    }

    function exitConversationMode() {
        state.conversationMode = false;
        logEvent("info", "conversation mode ended");
    }

    function bindComposer() {
        const form = $("composer");
        const input = $("composer-input");

        // Chrome gesture unlock
        const primeTts = () => {
            if (!window.speechSynthesis) return;
            try {
                const u = new SpeechSynthesisUtterance(" ");
                u.volume = 0;
                window.speechSynthesis.speak(u);
            } catch {}
        };
        const unlockOnce = () => {
            primeTts();
            // The user just interacted - arm the mic if we can.
            armMic();
            document.removeEventListener("click", unlockOnce, true);
            document.removeEventListener("keydown", unlockOnce, true);
        };
        document.addEventListener("click", unlockOnce, true);
        document.addEventListener("keydown", unlockOnce, true);

        form.addEventListener("submit", (ev) => {
            ev.preventDefault();
            const text = input.value.trim();
            if (!text) return;
            input.value = "";
            sendCommand(text);
        });

        $("send-btn").addEventListener("click", () => {
            form.dispatchEvent(new Event("submit", { cancelable: true }));
        });

        $("mute-btn").addEventListener("click", () => {
            state.ttsEnabled = !state.ttsEnabled;
            const btn = $("mute-btn");
            btn.dataset.on = state.ttsEnabled ? "1" : "0";
            btn.style.color = state.ttsEnabled ? "var(--cyan)" : "var(--text-dim)";
            btn.style.borderColor = state.ttsEnabled ? "var(--cyan)" : "var(--border-dim)";
            btn.title = state.ttsEnabled ? "Speech on (click to mute)" : "Speech muted (click to enable)";
            if (!state.ttsEnabled) stopSpeaking();
            logEvent("info", state.ttsEnabled ? "TTS enabled" : "TTS muted");
        });

        // Web Speech API
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            logEvent("warn", "Web Speech API unavailable - text only");
            $("mic-btn").addEventListener("click", () => {
                input.focus();
                logEvent("info", "voice input not supported in this browser");
            });
            return;
        }

        recognizer = new SR();
        recognizer.continuous = true;
        recognizer.interimResults = true;
        recognizer.lang = "en-US";

        const trigger = state.triggerWord.toLowerCase();

        recognizer.onstart = () => {
            recognizing = true;
            ensureAudioCtx(); // unlock on first user gesture
            logEvent("info", "mic ON — always listening for wake word");
            setListening(true);
        };

        recognizer.onresult = (ev) => {
            let finalTranscript = "";
            let interim = "";
            for (let i = ev.resultIndex; i < ev.results.length; i++) {
                const r = ev.results[i];
                if (r.isFinal) finalTranscript += r[0].transcript;
                else interim += r[0].transcript;
            }
            const combined = (finalTranscript || interim).toLowerCase().trim();
            if (!combined) return;

            // If we're in wake-mode, only react when we see the trigger word
            // anywhere in this chunk (final OR interim). Interim recognition is
            // critical for low-latency wake detection.
            if (!state.conversationMode) {
                const hit = _findTrigger(combined, trigger);
                if (hit !== null) {
                    enterConversationMode();
                    const remainder = combined.slice(hit + trigger.length).trim();
                    if (remainder.length > 1) {
                        logEvent("ok", `heard: ${remainder}`);
                        sendCommand(remainder);
                        exitConversationMode();
                    } else {
                        speak("Yes sir?");
                    }
                    // Clear the recognizer's buffer so the trigger word
                    // doesn't keep firing.
                    try { recognizer.abort(); recognizer.start(); } catch {}
                }
                return;
            }

            // Conversation mode: every final result is a command.
            if (finalTranscript) {
                const trimmed = finalTranscript.trim();
                if (trimmed.length > 0) {
                    logEvent("ok", `heard: ${trimmed}`);
                    sendCommand(trimmed);
                    state.conversationUntil = Date.now() + 8000;
                }
            }
        };

        recognizer.onerror = (ev) => {
            logEvent("warn", "mic: " + (ev.error || "unknown"));
            // not-allowed stops us trying forever
            if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
                wakeMode = false;
                recognizing = false;
                const mic = $("mic-btn");
                if (mic) {
                    mic.classList.remove("wake", "listening");
                    mic.title = "Mic permission denied — enable it in browser settings";
                }
            }
        };

        recognizer.onend = () => {
            // Auto-restart while wake mode is on. The browser often ends the
            // session silently after a few minutes of silence; we re-arm.
            if (wakeMode) {
                try {
                    recognizer.start();
                    return;
                } catch (e) {
                    setTimeout(() => { try { recognizer.start(); } catch {} }, 800);
                }
            }
            recognizing = false;
            setListening(false);
        };

        // Conversation timeout
        setInterval(() => {
            if (state.conversationMode && Date.now() > state.conversationUntil) {
                exitConversationMode();
            }
        }, 1000);

        // Mic button toggles the recognizer on/off (manual override).
        $("mic-btn").addEventListener("click", async () => {
            if (wakeMode && recognizing) {
                wakeMode = false;
                try { recognizer.stop(); } catch {}
                const mic = $("mic-btn");
                mic.classList.remove("wake", "listening");
                mic.title = "Mic OFF — click to enable";
                logEvent("info", "mic OFF");
                return;
            }
            await armMic();
        });
    }

    /**
     * Find the trigger word inside the transcript, allowing for one or two
     * STT misheard letters (e.g. "ervice" → "arvis"). Returns the index, or
     * null if not present.
     */
    function _findTrigger(haystack, needle) {
        const direct = haystack.indexOf(needle);
        if (direct !== -1) return direct;
        // Try common mishearings.
        const aliases = ["arvis", "ervice", "irvis", "arviss", "arevis", "travis"];
        let best = -1;
        for (const a of aliases) {
            const idx = haystack.indexOf(a);
            if (idx !== -1 && (best === -1 || idx < best)) best = idx;
        }
        return best === -1 ? null : best;
    }

    /**
     * Try to acquire the microphone and start the recognizer in wake mode.
     * Safe to call multiple times; subsequent calls after success are no-ops.
     */
    async function armMic() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            logEvent("warn", "Web Speech API unavailable - text only");
            return false;
        }
        if (!recognizer) {
            recognizer = new SR();
            recognizer.continuous = true;
            recognizer.interimResults = true;
            recognizer.lang = "en-US";
            // Recreate handlers when the recognizer is constructed late.
            // (We register handlers once in bindComposer; this is fine.)
        }
        try {
            if (!micStream) {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                await attachMicAnalyser(micStream);
            }
            wakeMode = true;
            const mic = $("mic-btn");
            mic.classList.add("wake");
            mic.title = "Mic ON — always listening for wake word";
            try { recognizer.start(); }
            catch (e) {
                // Already started - that's fine.
                if (!/already started/i.test(String(e && e.message))) {
                    throw e;
                }
            }
            logEvent("ok", "always-on mic armed");
            return true;
        } catch (err) {
            logEvent("err", "mic permission failed: " + err.message);
            return false;
        }
    }

    function setListening(on) {
        fetch("/api/listening", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ listening: !!on }),
        }).catch(() => {});
    }

    // -----------------------------------------------------------------
    // Quick actions (terminal / PC control)
    // -----------------------------------------------------------------
    function bindQuickActions() {
        qsa('[data-action="cmd"]').forEach((btn) => {
            btn.addEventListener("click", async () => {
                const cmd = btn.dataset.cmd || "";
                if (!cmd) return;
                btn.classList.add("active");
                setTimeout(() => btn.classList.remove("active"), 600);

                // Route by intent.
                if (cmd === "lock") {
                    const r = await fetch("/api/pc", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ command: "lock", confirm: true }),
                    }).then((r) => r.json()).catch(() => null);
                    logEvent(r && r.ok ? "ok" : "err", (r && r.message) || "PC control failed");
                    if (r && r.message) speak(r.message);
                    return;
                }

                if (/^volume\s+(up|down)$/.test(cmd) || cmd === "mute") {
                    const r = await fetch("/api/pc", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ command: cmd }),
                    }).then((r) => r.json()).catch(() => null);
                    logEvent(r && r.ok ? "ok" : "err", (r && r.message) || "PC control failed");
                    if (r && r.message) speak(r.message);
                    return;
                }

                // Run through the chat router so the LLM can speak the result.
                sendCommand(cmd);
            });
        });
    }

    // -----------------------------------------------------------------
    // Tabs (right pane)
    // -----------------------------------------------------------------
    function bindTabs() {
        const tabs = qsa(".tab", $("tabs"));
        tabs.forEach((t) => {
            t.addEventListener("click", () => {
                tabs.forEach((x) => x.classList.toggle("active", x === t));
                qsa(".tab-pane").forEach((p) => {
                    p.classList.toggle("active", p.dataset.tab === t.dataset.tab);
                });
            });
        });
    }

    // -----------------------------------------------------------------
    // Builder: custom commands / modes / reminders / Gmail
    // -----------------------------------------------------------------
    function bindBuilder() {
        $("reminder-add").addEventListener("click", async () => {
            const text = $("reminder-text").value.trim();
            const seconds = parseInt($("reminder-when").value || "0", 10);
            if (!text || !seconds) {
                logEvent("warn", "reminder: text and time are required");
                return;
            }
            const fireAt = Math.floor(Date.now() / 1000) + seconds;
            const r = await fetch("/api/reminders", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, fire_at: fireAt }),
            }).then((r) => r.json()).catch(() => null);
            if (r && r.ok) {
                $("reminder-text").value = "";
                logEvent("ok", `reminder set: ${text} (in ${seconds}s)`);
                speak(`Reminder set for ${seconds} seconds from now.`);
                await refreshReminders();
            }
        });

        $("ccmd-add").addEventListener("click", async () => {
            const name = $("ccmd-name").value.trim();
            const trigger = $("ccmd-trigger").value.trim();
            const response = $("ccmd-response").value.trim();
            if (!name || !trigger || !response) {
                logEvent("warn", "custom command: all fields required");
                return;
            }
            const r = await fetch("/api/custom/commands", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, trigger: [trigger], response }),
            }).then((r) => r.json()).catch(() => null);
            if (r && r.ok) {
                $("ccmd-name").value = "";
                $("ccmd-trigger").value = "";
                $("ccmd-response").value = "";
                logEvent("ok", `custom command '${name}' added`);
                await refreshCustomCommands();
            }
        });

        $("mode-add").addEventListener("click", async () => {
            const name = $("mode-name").value.trim();
            const colour = $("mode-colour").value.trim() || "#5fe4ff";
            const prompt = $("mode-prompt").value.trim();
            if (!name) {
                logEvent("warn", "custom mode: name required");
                return;
            }
            const r = await fetch("/api/custom/modes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, colour, prompt }),
            }).then((r) => r.json()).catch(() => null);
            if (r && r.ok) {
                $("mode-name").value = "";
                $("mode-prompt").value = "";
                logEvent("ok", `custom mode '${name}' added`);
                await refreshCustomModes();
            }
        });

        $("google-signin").addEventListener("click", async () => {
            const r = await fetch("/api/auth/google").then((r) => r.json()).catch(() => null);
            if (r && r.ok) {
                logEvent("info", "opening Google sign-in…");
                if (r.auth_url) window.open(r.auth_url, "_blank");
            } else {
                logEvent("err", "Google sign-in not configured");
                alert((r && r.error) || "Google OAuth is not configured on the server.");
            }
        });

        $("google-signout").addEventListener("click", async () => {
            await fetch("/api/auth/google/logout", { method: "POST" });
            await refreshGoogleStatus();
            logEvent("info", "signed out of Google");
        });
    }

    async function refreshCustomCommands() {
        const data = await fetch("/api/custom/commands").then((r) => r.json()).catch(() => null);
        if (!data) return;
        state.customCommands = data.commands || [];
        const ul = $("ccmds-list");
        ul.innerHTML = "";
        state.customCommands.forEach((c) => {
            const li = document.createElement("li");
            const trig = (c.trigger || []).join(", ") || "(no trigger)";
            li.innerHTML =
                `<div style="flex:1">` +
                    `<div>${escapeHtml(c.name)}</div>` +
                    `<div class="cmd-trig">${escapeHtml(trig)}</div>` +
                    `<div style="font-size:11px; color:var(--text-dim); margin-top:2px">${escapeHtml((c.response || "").slice(0, 80))}</div>` +
                `</div>`;
            const del = document.createElement("button");
            del.textContent = "DEL";
            del.addEventListener("click", async () => {
                await fetch(`/api/custom/commands/${encodeURIComponent(c.name)}`, { method: "DELETE" });
                await refreshCustomCommands();
            });
            li.appendChild(del);
            ul.appendChild(li);
        });
        $("ccmds-count").textContent = state.customCommands.length;
    }

    async function refreshCustomModes() {
        const data = await fetch("/api/custom/modes").then((r) => r.json()).catch(() => null);
        if (!data) return;
        state.customModes = (data.modes || []).map((m) => ({
            id: (m.name || "mode").toLowerCase(),
            name: m.name,
            colour: m.colour || "#5fe4ff",
            emoji: m.emoji || "◆",
            prompt: m.prompt || "",
        }));
        const ul = $("modes-builder-list");
        ul.innerHTML = "";
        state.customModes.forEach((m) => {
            const li = document.createElement("li");
            li.style.borderColor = m.colour;
            li.innerHTML =
                `<div style="flex:1">` +
                    `<div style="color:${escapeHtml(m.colour)}">${escapeHtml(m.name.toUpperCase())}</div>` +
                    `<div class="cmd-trig" style="color:${escapeHtml(m.colour)}">${escapeHtml((m.prompt || "").slice(0, 80))}</div>` +
                `</div>`;
            const del = document.createElement("button");
            del.textContent = "DEL";
            del.addEventListener("click", async () => {
                await fetch(`/api/custom/modes/${encodeURIComponent(m.name)}`, { method: "DELETE" });
                await refreshCustomModes();
                renderModes();
            });
            li.appendChild(del);
            ul.appendChild(li);
        });
        $("modes-count").textContent = state.customModes.length;
        renderModes();
    }

    async function refreshReminders() {
        const data = await fetch("/api/reminders").then((r) => r.json()).catch(() => null);
        if (!data) return;
        state.reminders = data.reminders || [];
        const ul = $("reminders-list");
        ul.innerHTML = "";
        state.reminders.forEach((r) => {
            const li = document.createElement("li");
            const when = new Date(r.fire_at * 1000).toLocaleString();
            li.innerHTML =
                `<div style="flex:1">` +
                    `<div>${escapeHtml(r.text)}</div>` +
                    `<div class="cmd-trig">${escapeHtml(when)}</div>` +
                `</div>`;
            const del = document.createElement("button");
            del.textContent = "DEL";
            del.addEventListener("click", async () => {
                await fetch(`/api/reminders/${encodeURIComponent(r.id)}`, { method: "DELETE" });
                await refreshReminders();
            });
            li.appendChild(del);
            ul.appendChild(li);
        });
        $("reminders-count").textContent = state.reminders.length;
    }

    async function refreshGoogleStatus() {
        const data = await fetch("/api/auth/google/status").then((r) => r.json()).catch(() => null);
        if (!data) return;
        state.google = {
            configured: !!data.configured,
            signed_in: !!data.signed_in,
            email: data.email || null,
        };
        $("google-configured").textContent = state.google.configured ? "YES" : "NO";
        $("google-signed").textContent = state.google.signed_in ? "YES" : "NO";
        $("google-email").textContent = state.google.email || "—";
        const pill = $("google-pill");
        if (state.google.signed_in) {
            pill.textContent = "GMAIL ●";
            pill.classList.add("pill-cyan");
            pill.style.color = "var(--cyan)";
        } else if (state.google.configured) {
            pill.textContent = "GMAIL ?";
            pill.style.color = "var(--gold)";
        } else {
            pill.textContent = "GMAIL ○";
            pill.classList.remove("pill-cyan");
            pill.style.color = "var(--text-dim)";
        }
        $("google-status-tag").textContent = state.google.signed_in ? "OK" : "…";
    }

    // -----------------------------------------------------------------
    // Bootstrap
    // -----------------------------------------------------------------
    async function bootstrap() {
        try {
            const resp = await fetch("/api/state");
            const data = await resp.json();
            state.history = data.chat || [];
            state.tools = data.tools || [];
            if (data.trigger) {
                state.triggerWord = data.trigger;
                $("trigger-pill").textContent = "WAKE: " + data.trigger;
            }
            state.customCommands = data.custom_commands || [];
            state.customModes = (data.custom_modes || []).map((m) => ({
                id: (m.name || "mode").toLowerCase(),
                name: m.name,
                colour: m.colour || "#5fe4ff",
                emoji: m.emoji || "◆",
                prompt: m.prompt || "",
            }));
            state.reminders = data.reminders || [];
            state.google = data.google || state.google;
            renderChat();
            renderTools();
            renderRecent();
            renderModes();
            renderRoutines();
            applyRuntimeState(data.runtime);
            applySystem(data.system);
        } catch (err) {
            logEvent("err", "bootstrap failed: " + err.message);
        }

        bindComposer();
        bindQuickActions();
        bindTabs();
        bindBuilder();
        $("clear-btn").addEventListener("click", async () => {
            await fetch("/api/clear", { method: "POST" });
            state.history = [];
            state.recent = [];
            renderChat();
            renderRecent();
            logEvent("info", "history cleared");
        });

        await refreshCustomCommands();
        await refreshCustomModes();
        await refreshReminders();
        await refreshGoogleStatus();
    }

    // -----------------------------------------------------------------
    // Polling fallback if Socket.IO client missing
    // -----------------------------------------------------------------
    function startPollingFallback() {
        logEvent("warn", "Socket.IO client not present - falling back to polling");
        setInterval(async () => {
            try {
                const r = await fetch("/api/system");
                applySystem(await r.json());
            } catch {}
        }, 2000);
    }

    function onReady() {
        bootstrap();
        if (typeof io === "function") {
            connectSocket();
        } else {
            startPollingFallback();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", onReady);
    } else {
        onReady();
    }
})();