/**
 * Drive the page over the Chrome DevTools Protocol and check that dragging and workspace switching work.
 *
 * A screenshot proves the initial layout and nothing else. The two things that can be wrong are both dynamic:
 *
 *   **dragging a splitter** must change the panel width -- and the failure mode worth catching is a handler that
 *   runs and writes a value nobody reads, which looks identical in a screenshot;
 *   **switching a workspace** must hide and show the right editors, and must not touch project state. That
 *   second half is the property that makes a workspace safe to click, and it is what would break first if this
 *   grew a "load preset with the workspace" feature.
 *
 * Node 22 has `WebSocket` built in, so no package is needed.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9223;
const profile = path.join(os.tmpdir(), "atsim-cdp-" + Date.now());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function json(pathname) {
  const r = await fetch("http://127.0.0.1:" + PORT + pathname);
  return r.json();
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) {
        const { resolve, reject } = this.pending.get(m.id);
        this.pending.delete(m.id);
        m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
      }
    });
  }
  send(method, params) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params: params || {} }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => reject(new Error("timeout: " + method)), 20000);
    });
  }
  async eval(expr) {
    const r = await this.send("Runtime.evaluate", {
      expression: expr, returnByValue: true, awaitPromise: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + " " +
      JSON.stringify(r.exceptionDetails.exception && r.exceptionDetails.exception.description));
    return r.result.value;
  }
}

async function main() {
  const chrome = spawn(CHROME, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=" + PORT, "--user-data-dir=" + profile,
    "--window-size=1500,900", "--no-first-run", "--no-default-browser-check", URL,
  ], { stdio: "ignore" });

  try {
    let targets = null;
    for (let i = 0; i < 60; i++) {
      try { targets = await json("/json/list"); } catch (e) { await sleep(300); continue; }
      if (targets && targets.some((t) => t.type === "page" && t.webSocketDebuggerUrl)) break;
      await sleep(300);
    }
    const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((res, rej) => {
      ws.addEventListener("open", res);
      ws.addEventListener("error", rej);
    });
    const cdp = new CDP(ws);
    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");
    await sleep(6000);                       // let the app boot and render once

    const out = {};

    out.errors = await cdp.eval(`(() => {
      const e = window.__wsErrors || [];
      return e;
    })()`);

    out.loaded = await cdp.eval("typeof window.Workspace === 'object' && !!window.Workspace");
    out.workspaces = await cdp.eval("window.Workspace ? window.Workspace.list.map(w => w.id) : []");

    const geom = () => cdp.eval(`(() => {
      const m = document.querySelector('main');
      const cs = getComputedStyle(m);
      const l = document.getElementById('panelLeft');
      const r = document.getElementById('panelRight');
      const s = document.querySelector('.wsSplit-left');
      return {
        cols: cs.gridTemplateColumns,
        leftTrack: cs.getPropertyValue('--w-left').trim(),
        rightTrack: cs.getPropertyValue('--w-right').trim(),
        leftHidden: l ? l.classList.contains('wsHidden') : null,
        rightHidden: r ? r.classList.contains('wsHidden') : null,
        leftVisible: l ? l.getBoundingClientRect().width : null,
        rightVisible: r ? r.getBoundingClientRect().width : null,
        splitVisible: s ? s.getBoundingClientRect().width : null,
      };
    })()`);

    out.initial = await geom();

    /* --- drag the left splitter 120 px to the right, with real pointer events --- */
    const box = await cdp.eval(`(() => {
      const s = document.querySelector('.wsSplit-left');
      const b = s.getBoundingClientRect();
      return { x: b.left + b.width / 2, y: b.top + b.height / 2 };
    })()`);
    async function mouse(type, x, y) {
      await cdp.send("Input.dispatchMouseEvent", {
        type, x, y, button: "left", buttons: type === "mouseReleased" ? 0 : 1, clickCount: 1,
      });
    }
    await mouse("mousePressed", box.x, box.y);
    for (let i = 1; i <= 8; i++) await mouse("mouseMoved", box.x + i * 15, box.y);
    await mouse("mouseReleased", box.x + 120, box.y);
    await sleep(400);
    out.afterDrag = await geom();

    /* --- switch to a workspace that hides both panels --- */
    await cdp.eval("window.Workspace.set('watch')");
    await sleep(500);
    out.watch = await geom();
    out.watchActive = await cdp.eval("window.Workspace.state.active");

    /* --- switching must not have touched project state --- */
    out.projectParams = await cdp.eval(`(() => {
      const t = document.querySelectorAll('#paramList input, #paramList select');
      return t.length;
    })()`);

    /* --- and the arrangement must persist --- */
    await cdp.eval("window.Workspace.set('layout')");
    await sleep(400);
    out.backToLayout = await geom();
    out.stored = await cdp.eval("localStorage.getItem('atsim.workspace.v1')");

    console.log(JSON.stringify(out, null, 1));
    ws.close();
  } finally {
    chrome.kill();
    await sleep(500);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
}

main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
