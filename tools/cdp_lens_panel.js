/**
 * Confirm the new parameters are on screen, in the real page, by asking the DOM.
 *
 * "It is in the schema" and "it is on screen" are different claims and this project has confused them before -- a
 * button that was appended to the right row of code that never ran. So the check is the rendered label text, and
 * the tab that has to be open for it to exist.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9231;
const profile = path.join(os.tmpdir(), "atsim-lens-" + Date.now());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const WANT = ["光轴偏心", "像场弯曲", "纵向色差", "彗差", "像散", "球差", "镜头畸变", "横向色差"];

async function main() {
  const chrome = spawn(CHROME, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=" + PORT, "--user-data-dir=" + profile,
    "--window-size=1500,950", "--no-first-run", "--no-default-browser-check", URL,
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
    await sleep(8000);

    // open the camera tab in the parameter panel, which is where the lens group lives
    const out = await evaluate(`(() => {
      const tabs = Array.from(document.querySelectorAll('#layerTabs .tab, #layerTabs button'));
      const cam = tabs.find((t) => /\\u6444\\u50cf\\u673a/.test(t.textContent || ''));
      if (cam) cam.click();
      return { tabs: tabs.map((t) => (t.textContent || '').trim()), clicked: !!cam };
    })()`);
    await sleep(1200);

    const probe = await evaluate(`(() => {
      const list = document.getElementById('paramList');
      if (!list) return { error: 'no #paramList' };
      const text = list.textContent || '';
      const labels = Array.from(list.querySelectorAll('label, .paramLabel, .plabel, .pname'))
        .map((e) => (e.textContent || '').trim());
      const groups = Array.from(list.querySelectorAll('.palGroup, .groupHead, .pgroup'))
        .map((e) => (e.textContent || '').trim());
      const want = ${JSON.stringify(WANT)};
      return {
        groupCount: groups.length,
        groups: groups.slice(0, 12),
        labelCount: labels.length,
        found: want.filter((w) => text.indexOf(w) >= 0),
        missing: want.filter((w) => text.indexOf(w) < 0),
        scrollHeight: list.scrollHeight,
      };
    })()`);
    console.log(JSON.stringify({ tabs: out, panel: probe }, null, 1));

    const shot = await send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(path.join("tools", "ntscrs_ref", "lens_panel.png"),
      Buffer.from(shot.data, "base64"));
    console.log("wrote lens_panel.png");
    ws.close();
  } finally {
    chrome.kill();
    await sleep(400);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
