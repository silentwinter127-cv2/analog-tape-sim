/**
 * Enumerate every button the page actually renders inside the subtitle panel, with its real geometry.
 *
 * The previous probe looked at `#osdBody`, which does not exist -- the builder mounts into `#osdPanel`, and
 * `#osdBody` is an artifact of a different design that was never used. So it reported "no buttons" while
 * reporting the panel's text in the same breath, which is the contradiction that wasted several turns.
 *
 * This lists what is there: tag, text, class, position, size, computed display, and whether it is inside the
 * pinned row. And it opens the subtitle tab first, because a hidden panel's buttons are not rendered and asking
 * about them before opening it is how the last five reports went wrong.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9227;
const profile = path.join(os.tmpdir(), "atsim-btns-" + Date.now());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const chrome = spawn(CHROME, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=" + PORT, "--user-data-dir=" + profile,
    "--window-size=1500,900", "--no-first-run", "--no-default-browser-check", URL,
  ], { stdio: "ignore" });
  try {
    let targets = null;
    for (let i = 0; i < 60; i++) {
      try { targets = await (await fetch("http://127.0.0.1:" + PORT + "/json/list")).json(); }
      catch (e) { await sleep(300); continue; }
      if (targets && targets.some((t) => t.type === "page" && t.webSocketDebuggerUrl)) break;
      await sleep(300);
    }
    const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.addEventListener("open", res); ws.addEventListener("error", rej); });
    let id = 0;
    const pending = new Map();
    const logs = [];
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
      if (m.method === "Runtime.consoleAPICalled") {
        logs.push(m.params.type + ": " + (m.params.args || []).map((a) => a.value ||
          (a.description || "")).join(" ").slice(0, 300));
      }
      if (m.method === "Runtime.exceptionThrown") {
        logs.push("EXCEPTION: " + JSON.stringify(m.params.exceptionDetails).slice(0, 300));
      }
    });
    const send = (method, params) => new Promise((res) => {
      const i = ++id; pending.set(i, res);
      ws.send(JSON.stringify({ id: i, method, params: params || {} }));
    });
    const evaluate = async (expression) => {
      const r = await send("Runtime.evaluate", { expression, returnByValue: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + " " +
        (r.exceptionDetails.exception && r.exceptionDetails.exception.description));
      return r.result.value;
    };
    await send("Runtime.enable");
    await sleep(7000);

    // open the subtitle tab, exactly as a person would
    await evaluate(`(() => {
      const tabs = Array.from(document.querySelectorAll('#ssTabs .tab'));
      const t = tabs.find((b) => (b.textContent || '').indexOf('\\u5b57\\u5e55') >= 0);
      if (t) t.click();
      return tabs.map((b) => (b.textContent || '').trim());
    })()`);
    await sleep(600);

    console.log(JSON.stringify(await evaluate(`(() => {
      const panel = document.getElementById('osdPanel');
      const sticky = document.querySelector('.osdSticky');
      const list = (root) => Array.from(root.querySelectorAll('button')).map((b) => {
        const r = b.getBoundingClientRect();
        const cs = getComputedStyle(b);
        return {
          text: (b.textContent || '').trim().slice(0, 18),
          cls: b.className,
          w: Math.round(r.width), h: Math.round(r.height),
          x: Math.round(r.x), y: Math.round(r.y),
          display: cs.display,
          visible: r.width > 0 && r.height > 0 && cs.display !== 'none',
        };
      });
      return {
        panelHidden: panel.classList.contains('hidden'),
        panelChildren: panel.children.length,
        stickyChildren: sticky ? sticky.children.length : -1,
        stickyTags: sticky ? Array.from(sticky.children).map((c) => c.tagName + '.' + c.className) : [],
        buttonsInPanel: list(panel),
        buttonsInSticky: sticky ? list(sticky) : [],
        panelScrollH: panel.scrollHeight,
        panelClientH: panel.clientHeight,
        panelScrollTop: panel.scrollTop,
      };
    })()`), null, 1));

    // scroll the panel to the bottom, to see whether the button is merely below the fold after all
    await evaluate(`(() => { const p = document.getElementById('osdPanel');
      p.scrollTop = p.scrollHeight; return p.scrollTop; })()`);
    await sleep(400);
    console.log("after scrolling to the bottom:");
    console.log(JSON.stringify(await evaluate(`(() => {
      const sticky = document.querySelector('.osdSticky');
      const p = document.getElementById('osdPanel');
      return { scrollTop: p.scrollTop, stickyTop: Math.round(sticky.getBoundingClientRect().top),
               stickyChildren: sticky.children.length };
    })()`)));

    const shot = await send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(path.join("tools", "ntscrs_ref", "buttons_open.png"),
      Buffer.from(shot.data, "base64"));
    console.log("console/log events: " + JSON.stringify(logs.slice(0, 6)));
    ws.close();
  } finally {
    chrome.kill();
    await sleep(400);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
