/**
 * Screenshot the page in two workspaces, which is the only proof that the arrangement is right.
 *
 * The computed `grid-template-columns` keeps reporting the named tracks even when nothing occupies them, so
 * `0px 5px 1474px 5px 0px` is not evidence either way. What matters is where the stage actually begins and ends,
 * and a screen capture is the direct answer. It also catches what a measurement would miss: a splitter line left
 * behind, a panel that is hidden but still painted, or the workspace bar overlapping the picture.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9224;
const profile = path.join(os.tmpdir(), "atsim-shot-" + Date.now());
const OUT = path.join("tools", "ntscrs_ref");
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
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
    });
    const send = (method, params) => new Promise((res) => {
      const i = ++id; pending.set(i, res);
      ws.send(JSON.stringify({ id: i, method, params: params || {} }));
    });
    const evaluate = async (expression) => {
      const r = await send("Runtime.evaluate", { expression, returnByValue: true });
      return r.result && r.result.value;
    };
    await send("Runtime.enable");
    await sleep(7000);

    const geometry = () => evaluate(`(() => {
      const st = document.querySelector('.stage').getBoundingClientRect();
      const l = document.getElementById('panelLeft').getBoundingClientRect();
      return { stageLeft: Math.round(st.left), stageRight: Math.round(st.right),
               stageWidth: Math.round(st.width),
               leftWidth: Math.round(l.width), win: window.innerWidth,
               barHeight: Math.round(document.querySelector('.wsBar').getBoundingClientRect().height) };
    })()`);

    for (const w of ["layout", "watch", "model", "grade"]) {
      await evaluate(`window.Workspace.set(${JSON.stringify(w)})`);
      await sleep(1200);
      const g = await geometry();
      const shot = await send("Page.captureScreenshot", { format: "png" });
      fs.writeFileSync(path.join(OUT, "ws_" + w + ".png"), Buffer.from(shot.data, "base64"));
      console.log(JSON.stringify({ workspace: w, ...g }));
    }
    ws.close();
  } finally {
    chrome.kill();
    await sleep(400);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
