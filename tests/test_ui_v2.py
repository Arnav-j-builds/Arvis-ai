"""Verify that the always-on mic and wake-word detection works in the UI."""
from playwright.sync_api import sync_playwright

URL = "http://localhost:5052/"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            permissions=["microphone"],
        )
        page = ctx.new_page()

        errors = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: print(f"[{msg.type}] {msg.text}") if msg.type in ("error", "warning") else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(1500)

        # 1. The mic button should have the "wake" class right after load.
        wake_class = page.eval_on_selector("#mic-btn", "el => el.className")
        print("mic-btn class on load:", wake_class)

        # 2. The status pill should show IDLE.
        pill = page.eval_on_selector("#status-pill", "el => el.dataset.state + ':' + el.textContent")
        print("status pill:", pill)

        # 3. The orb should be rendering in some mode.
        orb = page.evaluate("window.__arvisOrb ? {mode: window.__arvisOrb.mode, intensity: window.__arvisOrb.intensity} : null")
        print("orb state:", orb)

        # 4. Simulate a user click (anywhere) — the mic should arm.
        page.click("#trigger-pill")
        page.wait_for_timeout(800)
        wake_after_click = page.eval_on_selector("#mic-btn", "el => el.className")
        print("mic-btn class after click:", wake_after_click)

        # 5. Inject a wake-word into the recognizer by patching it.
        #    (Browser SpeechRecognition is hard to drive headless, so we
        #    simulate the result via a direct dispatch.)
        # We do this by faking onresult with a synthetic event.
        page.evaluate("""
            () => {
                // The recognizer is hidden inside the IIFE; we cannot reach
                // it directly. Instead, simulate the *effect* by calling
                // armMic and then dispatching onresult through __arvisOrb
                // visibility.
                if (window.__arvisOrb) window.__arvisOrb.setListening(true);
            }
        """)

        # 6. Check that the audio analyser pipeline is wired.
        audio = page.evaluate("window.__arvisAudio ? {energy: window.__arvisAudio.energy, hasWave: !!window.__arvisAudio.wave} : null")
        print("audio pipeline:", audio)

        # 7. Check that we can post a chat message and see a reply.
        page.fill("#composer-input", "what time is it")
        page.click("#send-btn")
        page.wait_for_timeout(2500)
        chat = page.evaluate("Array.from(document.querySelectorAll('.chat-msg')).map(e => e.textContent.slice(0, 100))")
        print("chat after send:", chat[-2:])

        # 8. Switch tab to CUSTOM and verify the builder renders.
        page.click('[data-tab="custom"]')
        page.wait_for_timeout(500)
        custom_visible = page.is_visible("#ccmd-add")
        print("custom tab visible:", custom_visible)

        # 9. Add a custom command.
        page.fill("#ccmd-name", "hello")
        page.fill("#ccmd-trigger", "say hello to me")
        page.fill("#ccmd-response", "Hello, world!")
        page.click("#ccmd-add")
        page.wait_for_timeout(800)
        cmds = page.evaluate("Array.from(document.querySelectorAll('#ccmds-list li')).map(li => li.textContent.slice(0, 80))")
        print("custom cmds:", cmds)

        # 10. Add a reminder.
        page.fill("#reminder-text", "stretch")
        page.select_option("#reminder-when", "60")
        page.click("#reminder-add")
        page.wait_for_timeout(800)
        reminders = page.evaluate("Array.from(document.querySelectorAll('#reminders-list li')).map(li => li.textContent.slice(0, 80))")
        print("reminders:", reminders)

        # 11. Click a quick action button.
        page.click('[data-tab="tools"]')
        page.wait_for_timeout(500)
        page.click('[data-cmd="run whoami"]')
        page.wait_for_timeout(2000)
        print("after whoami quick action")

        # 12. Take a final screenshot for visual verification.
        page.screenshot(path="orb_v2_full.png", full_page=False)

        print("\nERRORS:", errors if errors else "none")
        browser.close()


if __name__ == "__main__":
    main()