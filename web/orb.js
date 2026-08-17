/* =====================================================================
   Arvis 3D Orb - ALIVE edition
   =====================================================================

   The orb is alive:
   * WebGL preferred, 2D canvas guaranteed fallback.
   * Breathing pulse when idle.
   * Audio-reactive (mic + speaker amplitude via AnalyserNode).
   * Mode-aware colour: switchable core hue + glow.
   * Ripple rings while assistant is speaking.
   * Pointer parallax.
   ===================================================================== */

(function () {
    "use strict";

    function log(...args) {
        // eslint-disable-next-line no-console
        console.log("[arvis-orb]", ...args);
    }

    const canvas = document.getElementById("orb-canvas");
    if (!canvas) {
        log("ERROR: #orb-canvas not found in DOM");
        return;
    }
    log("canvas found", canvas);

    // -----------------------------------------------------------------
    // 2D Canvas fallback
    // -----------------------------------------------------------------
    function drawOrb2D(ctx, w, h, t, audio) {
        const cx = w / 2;
        const cy = h / 2;
        const minDim = Math.min(w, h);
        const state = window.__arvisOrb || { intensity: 0.6, hue: 195, listening: false, thinking: false };

        // Audio energy 0..1
        const energy = (audio && audio.energy) || 0;
        const bass = (audio && audio.bass) || 0;
        const hue = state.hue || 195;
        const baseColor = `hsl(${hue}, 100%, 65%)`;
        const accentColor = `hsl(${(hue + 30) % 360}, 100%, 70%)`;
        const deepColor = `hsl(${hue}, 80%, 30%)`;

        // Background nebula
        const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, minDim * 0.6);
        bg.addColorStop(0, `hsla(${hue}, 100%, 70%, 0.18)`);
        bg.addColorStop(0.5, `hsla(${hue}, 100%, 60%, 0.04)`);
        bg.addColorStop(1, "rgba(4, 6, 26, 0)");
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);

        // Breathing scale
        const breathe = 0.96 + 0.04 * Math.sin(t * (state.listening ? 6 : 1.4));
        const intensity = (state.intensity || 0.6) + energy * 0.4;
        const baseR = minDim * 0.20 * breathe;

        // Concentric orbital rings (audio-modulated)
        const rings = [
            { r: baseR * 1.55, color: baseColor, a: 0.65, width: 2.0, tilt: 0.0,  speed: 0.10 },
            { r: baseR * 1.78, color: accentColor, a: 0.45, width: 1.2, tilt: 0.3, speed: 0.14 },
            { r: baseR * 2.00, color: baseColor,   a: 0.30, width: 0.8, tilt: -0.25, speed: 0.18 },
            { r: baseR * 2.25, color: baseColor,   a: 0.18, width: 0.6, tilt: 0.4, speed: 0.22 },
        ];
        rings.forEach((ring, i) => {
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(t * ring.speed * (i % 2 ? 1 : -1));
            ctx.scale(1, Math.cos(ring.tilt + t * 0.1) * (1 + bass * 0.15));
            ctx.strokeStyle = ring.color.replace(")", `,${ring.a})`).replace("hsl", "hsla");
            ctx.lineWidth = ring.width * (1 + bass * 0.5);
            ctx.shadowColor = baseColor;
            ctx.shadowBlur = 8 + bass * 12;
            ctx.beginPath();
            ctx.arc(0, 0, ring.r, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        });

        // Core sphere
        const r = baseR * (1.0 + (intensity - 0.5) * 0.10 + energy * 0.05);
        const core = ctx.createRadialGradient(
            cx - r * 0.3, cy - r * 0.3, 0,
            cx, cy, r,
        );
        core.addColorStop(0,   `rgba(255, 255, 255, ${0.95 * intensity})`);
        core.addColorStop(0.3, `hsla(${hue}, 100%, 70%, ${0.85 * intensity})`);
        core.addColorStop(0.7, deepColor.replace(")", `,${0.85 * intensity})`).replace("hsl", "hsla"));
        core.addColorStop(1,   "rgba(4, 6, 26, 0.95)");
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();

        // Wireframe icosahedron approximation
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(t * 0.18);
        ctx.strokeStyle = `hsla(${hue}, 100%, 70%, ${0.65 * intensity})`;
        ctx.lineWidth = 1.2;
        ctx.shadowColor = baseColor;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(0, 0, r * 1.05, 0, Math.PI * 2);
        ctx.stroke();
        for (let i = 0; i < 3; i++) {
            ctx.save();
            ctx.rotate((Math.PI * 2 / 3) * i);
            ctx.beginPath();
            ctx.ellipse(0, 0, r * 1.05, r * 0.35, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
        ctx.beginPath();
        ctx.ellipse(0, 0, r * 0.35, r * 1.05, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Inner hotspot
        const hot = ctx.createRadialGradient(
            cx - r * 0.35, cy - r * 0.35, 0,
            cx - r * 0.35, cy - r * 0.35, r * 0.5,
        );
        hot.addColorStop(0, "rgba(255, 255, 255, 0.85)");
        hot.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = hot;
        ctx.beginPath();
        ctx.arc(cx - r * 0.35, cy - r * 0.35, r * 0.5, 0, Math.PI * 2);
        ctx.fill();

        // Particle field (orbit-drift)
        const particles = window.__arvisOrb && window.__arvisOrb._particles;
        if (particles) {
            for (const p of particles) {
                const px = cx + p.x * minDim * 0.35;
                const py = cy + p.y * minDim * 0.35;
                ctx.fillStyle = `hsla(${hue}, 100%, 75%, ${p.a})`;
                ctx.beginPath();
                ctx.arc(px, py, 1.4 + bass * 1.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Audio waveform halo (when energy is meaningful)
        if (energy > 0.02 && audio && audio.wave) {
            ctx.save();
            ctx.translate(cx, cy);
            ctx.strokeStyle = `hsla(${hue}, 100%, 80%, 0.6)`;
            ctx.lineWidth = 1.4;
            ctx.shadowColor = baseColor;
            ctx.shadowBlur = 10;
            ctx.beginPath();
            for (let i = 0; i < audio.wave.length; i++) {
                const a = (i / audio.wave.length) * Math.PI * 2;
                const amp = audio.wave[i] * (baseR * 1.6);
                const x = Math.cos(a) * (baseR * 2.4 + amp);
                const y = Math.sin(a) * (baseR * 2.4 + amp);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.restore();
        }
    }

    function makeParticles() {
        const out = [];
        const n = 110;
        for (let i = 0; i < n; i++) {
            const ang = Math.random() * Math.PI * 2;
            const dist = 0.4 + Math.random() * 1.4;
            out.push({
                x: Math.cos(ang) * dist,
                y: Math.sin(ang) * dist * 0.85,
                a: 0.3 + Math.random() * 0.6,
                vx: (Math.random() - 0.5) * 0.0008,
                vy: (Math.random() - 0.5) * 0.0008,
            });
        }
        return out;
    }

    // -----------------------------------------------------------------
    // Shared state
    // -----------------------------------------------------------------
    const orbState = {
        listening: false,
        thinking: false,
        speaking: false,
        intensity: 0.6,
        hue: 195,                       // 195 = cyan; user can override
        mode: "pending",
        setListening(v) {
            this.listening = !!v;
            this.intensity = v ? 1.0 : (this.thinking ? 0.85 : (this.speaking ? 0.95 : 0.6));
            document.body && document.body.setAttribute("data-wake", v ? "on" : "off");
        },
        setThinking(v) {
            this.thinking = !!v;
            this.intensity = v ? 0.85 : (this.listening ? 1.0 : (this.speaking ? 0.95 : 0.6));
        },
        setSpeaking(v) {
            this.speaking = !!v;
            this.intensity = v ? 0.95 : (this.listening ? 1.0 : (this.thinking ? 0.85 : 0.6));
            document.body && document.body.setAttribute("data-speaking", v ? "on" : "off");
        },
        setHue(h) {
            if (typeof h === "number") this.hue = h;
        },
        _particles: makeParticles(),
    };
    window.__arvisOrb = orbState;
    log("orb state initialised", orbState);

    // -----------------------------------------------------------------
    // Renderer resolver
    // -----------------------------------------------------------------
    function tryWebGL() {
        if (!window.THREE) return false;
        try {
            const test = document.createElement("canvas");
            const gl = test.getContext("webgl2") || test.getContext("webgl");
            return !!gl;
        } catch (e) { return false; }
    }

    const useWebGL = tryWebGL();

    // -----------------------------------------------------------------
    // 2D renderer
    // -----------------------------------------------------------------
    function start2D() {
        orbState.mode = "2d";
        log("starting 2D renderer");
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            log("ERROR: 2D context unavailable");
            return;
        }
        const start = performance.now();
        let lastFrameLog = 0;

        function frame() {
            for (const p of orbState._particles) {
                p.x += p.vx;
                p.y += p.vy;
                if (Math.abs(p.x) > 1.5) p.vx *= -1;
                if (Math.abs(p.y) > 1.5) p.vy *= -1;
            }

            let w = canvas.clientWidth;
            let h = canvas.clientHeight;
            if (!w || !h) {
                const rect = canvas.getBoundingClientRect();
                w = rect.width || canvas.width || 800;
                h = rect.height || canvas.height || 480;
            }
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const bw = Math.floor(w * dpr);
            const bh = Math.floor(h * dpr);
            if (canvas.width !== bw || canvas.height !== bh) {
                canvas.width = bw;
                canvas.height = bh;
            }
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            drawOrb2D(ctx, w, h, (performance.now() - start) / 1000, window.__arvisAudio || null);

            const now = performance.now();
            if (now - lastFrameLog > 2000) {
                log("2D frame alive", w, "x", h, "state", orbState.mode);
                lastFrameLog = now;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    // -----------------------------------------------------------------
    // WebGL renderer
    // -----------------------------------------------------------------
    function startWebGL() {
        orbState.mode = "webgl";
        log("starting WebGL renderer");
        const THREE = window.THREE;
        const scene = new THREE.Scene();
        scene.background = null;
        scene.fog = new THREE.FogExp2(0x04061a, 0.012);

        let renderer;
        try {
            renderer = new THREE.WebGLRenderer({
                canvas, antialias: true, alpha: true,
                powerPreference: "high-performance",
            });
        } catch (err) {
            log("WebGLRenderer ctor failed:", err.message, "- falling back to 2D");
            return start2D();
        }
        log("renderer created", renderer.capabilities.isWebGL2 ? "(WebGL2)" : "(WebGL1)");

        const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
        camera.position.set(0, 0, 14);

        const coreGroup = new THREE.Group();
        scene.add(coreGroup);

        // Wireframe inner
        const wireGeo = new THREE.IcosahedronGeometry(2.5, 1);
        const wireMat = new THREE.LineBasicMaterial({
            color: 0x5fe4ff, transparent: true, opacity: 0.85,
        });
        const wireMesh = new THREE.LineSegments(new THREE.WireframeGeometry(wireGeo), wireMat);
        coreGroup.add(wireMesh);

        // Wireframe outer
        const shellMesh = new THREE.LineSegments(
            new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(3.0, 1)),
            new THREE.LineBasicMaterial({ color: 0x7fdfff, transparent: true, opacity: 0.30 }),
        );
        coreGroup.add(shellMesh);

        const ringsGroup = new THREE.Group();
        scene.add(ringsGroup);

        function makeRing(radius, color, opacity, tubeRadius, axis) {
            const geo = new THREE.TorusGeometry(radius, tubeRadius, 16, 128);
            const mat = new THREE.MeshBasicMaterial({
                color, transparent: true, opacity,
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const ring = new THREE.Mesh(geo, mat);
            if (axis) {
                ring.rotation.x = axis.x;
                ring.rotation.y = axis.y;
                ring.rotation.z = axis.z;
            }
            return ring;
        }
        ringsGroup.add(makeRing(3.8, 0x5fe4ff, 0.65, 0.020, new THREE.Euler(Math.PI / 2, 0, 0)));
        ringsGroup.add(makeRing(4.3, 0x7fdfff, 0.45, 0.012, new THREE.Euler(Math.PI / 2 + 0.3, 0.4, 0)));
        ringsGroup.add(makeRing(4.8, 0x9ff0ff, 0.30, 0.008, new THREE.Euler(Math.PI / 2 - 0.25, -0.3, 0.2)));
        ringsGroup.add(makeRing(5.4, 0x5fe4ff, 0.18, 0.006, new THREE.Euler(0.4, 0.6, 0)));

        function makeParticles(count, radius, size, color, opacity) {
            const positions = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                const r = radius * (0.4 + Math.random() * 0.9);
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos((Math.random() * 2) - 1);
                positions[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta);
                positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
                positions[i * 3 + 2] = r * Math.cos(phi);
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            return new THREE.Points(geo, new THREE.PointsMaterial({
                color, size, transparent: true, opacity,
                blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
            }));
        }
        scene.add(makeParticles(380, 8.0, 0.045, 0x5fe4ff, 0.85));
        scene.add(makeParticles(220, 14.0, 0.025, 0x9ff0ff, 0.55));
        scene.add(makeParticles(120, 22.0, 0.018, 0x7fdfff, 0.30));

        const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
        canvas.addEventListener("pointermove", (ev) => {
            const rect = canvas.getBoundingClientRect();
            pointer.tx = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.ty = ((ev.clientY - rect.top) / rect.height) * 2 - 1;
        });
        canvas.addEventListener("pointerleave", () => { pointer.tx = 0; pointer.ty = 0; });

        const clock = new THREE.Clock();

        function resize() {
            let w = canvas.clientWidth, h = canvas.clientHeight;
            if (!w || !h) {
                const rect = canvas.getBoundingClientRect();
                w = rect.width; h = rect.height;
            }
            if (!w || !h) { w = 800; h = 480; }
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }
        requestAnimationFrame(resize);
        window.addEventListener("resize", resize);
        if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);

        function tick() {
            const t = clock.getElapsedTime();
            pointer.x += (pointer.tx - pointer.x) * 0.05;
            pointer.y += (pointer.ty - pointer.y) * 0.05;
            const energy = ((window.__arvisAudio && window.__arvisAudio.energy) || 0);
            wireMesh.rotation.x = t * 0.18 + pointer.y * 0.4;
            wireMesh.rotation.y = t * 0.24 + pointer.x * 0.4;
            shellMesh.rotation.x = -t * 0.10;
            shellMesh.rotation.y =  t * 0.14;
            coreGroup.rotation.y  =  t * 0.06;
            ringsGroup.children.forEach((ring, i) => {
                ring.rotation.z = t * (0.10 + i * 0.04) * (i % 2 ? 1 : -1);
            });
            const pulse = 0.85 + 0.15 * Math.sin(t * (orbState.listening ? 6 : 2));
            wireMat.opacity = 0.55 + 0.45 * orbState.intensity * pulse;
            shellMesh.material.opacity = 0.18 + 0.25 * orbState.intensity;
            const s = 1.0 + (orbState.intensity - 0.5) * 0.10 + energy * 0.10
                + (orbState.listening ? Math.sin(t * 8) * 0.02 : 0);
            coreGroup.scale.setScalar(s);
            camera.position.x = pointer.x * 1.5;
            camera.position.y = -pointer.y * 1.5;
            camera.lookAt(0, 0, 0);
            renderer.render(scene, camera);
            requestAnimationFrame(tick);
        }
        tick();
    }

    // -----------------------------------------------------------------
    // Static boot stamp
    // -----------------------------------------------------------------
    (function stampCanvas() {
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const w = canvas.width || canvas.clientWidth || 800;
        const h = canvas.height || canvas.clientHeight || 480;
        ctx.fillStyle = "rgba(95, 228, 255, 0.5)";
        ctx.font = "16px Orbitron, monospace";
        ctx.textAlign = "center";
        ctx.fillText("ARVIS // booting orb…", w / 2, h / 2);
    })();

    // -----------------------------------------------------------------
    // KICK OFF
    // -----------------------------------------------------------------
    if (useWebGL) {
        try {
            startWebGL();
        } catch (err) {
            log("WebGL start threw:", err.message, "- falling back to 2D");
            start2D();
        }
    } else {
        start2D();
    }

    setTimeout(() => {
        if (orbState.mode === "pending") {
            log("no frame logged after 2s - forcing 2D fallback");
            start2D();
        }
    }, 2000);
})();

// =====================================================================
// Background starfield
// =====================================================================
(function () {
    const bgCanvas = document.getElementById("bg-particles");
    if (!bgCanvas) return;
    const ctx = bgCanvas.getContext("2d");
    if (!ctx) return;
    let stars = [];

    function resize() {
        const dpr = Math.min(window.devicePixelRatio, 2);
        bgCanvas.width = window.innerWidth * dpr;
        bgCanvas.height = window.innerHeight * dpr;
        bgCanvas.style.width = window.innerWidth + "px";
        bgCanvas.style.height = window.innerHeight + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        stars = [];
        for (let i = 0; i < 180; i++) {
            stars.push({
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                r: Math.random() * 1.4 + 0.2,
                a: Math.random() * 0.6 + 0.2,
                tw: Math.random() * 0.02 + 0.005,
                phase: Math.random() * Math.PI * 2,
            });
        }
    }
    resize();
    window.addEventListener("resize", resize);

    function draw(t) {
        ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
        for (const s of stars) {
            const a = s.a * (0.6 + 0.4 * Math.sin(t * s.tw + s.phase));
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(160, 220, 255, ${a})`;
            ctx.fill();
        }
    }

    let t0 = performance.now();
    (function loop() {
        draw((performance.now() - t0) / 1000);
        requestAnimationFrame(loop);
    })();
})();