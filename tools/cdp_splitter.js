/**
 * Check that the splitter is confined to the editor row and does not overlap the workspace bar.
 *
 * The user's screenshot showed an orange stub under the subtitle panel, which was the splitter drawn across the
 * bar's row: `top: 0` with `main` as the containing block, and `main` spans both grid rows. Every measurement I
 * had taken was of the splitter's *width* and *x*, both of which were correct -- so this asserts the one
 * property that was wrong, by comparing the splitter's top against the bar's bottom. Two rectangles agreeing is
 * the whole test; no pixel needs looking at.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9225;
const profile = path.join(os.tmpdir(), "atsim-spl-" + Date.now());
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
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
      return r.result.value;
    };
    await send("Runtime.enable");
    await sleep(7000);

    const geometry = () => evaluate(`(() => {
      const bar = document.querySelector('.wsBar').getBoundingClientRect();
      const stage = document.querySelector('.stage').getBoundingClientRect();
      const l = document.querySelector('.wsSplit-left').getBoundingClientRect();
      const r = document.querySelector('.wsSplit-right').getBoundingClientRect();
      return {
        barTop: Math.round(bar.top), barBottom: Math.round(bar.bottom), barH: Math.round(bar.height),
        stageTop: Math.round(stage.top),
        leftTop: Math.round(l.top), leftBottom: Math.round(l.bottom),
        rightTop: Math.round(r.top), rightBottom: Math.round(r.bottom),
        winH: window.innerHeight,
      };
    })()`);

    for (const w of ["layout", "watch"]) {
      await evaluate(`window.Workspace.set(${JSON.stringify(w)})`);
      await sleep(900);
      const g = await geometry();
      const ok = g.leftTop >= g.barBottom - 1 && g.rightTop >= g.barBottom - 1;
      const reachesBottom = Math.abs(g.leftBottom - (g.winH - 108 + 40)) < 60;
      console.log(JSON.stringify({
        workspace: w, ...g,
        startsBelowTheBar: ok,
        verdict: ok ? "confined to the editor row" : "OVERLAPS THE BAR",
      }));
      const shot = await send("Page.captureScreenshot", { format: "png" });
      fs.writeFileSync(path.join("tools", "ntscrs_ref", "spl_" + w + ".png"),
        Buffer.from(shot.data, "base64"));
    }
    ws.close();
  } finally {
    chrome.kill();
    await sleep(400);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
