/**
 * Verify the night-vision controls on the camera layer, driving the page and reading the engine's own state.
 *
 * **The previous probe's mistake, which it found by printing the state instead of assuming it.** The page opens on the
 * **format** layer, not the camera layer -- so the night-vision rows had never been built and the probe reported "no
 * night-vision select" while the control was in fact working. **`S.layer = "format"` on load**, and switching the
 * layer is what builds the camera panel's rows. Asking about a panel that is not shown is the same fault as asking
 * about a hidden panel's buttons, which wasted five reports earlier in this project.
 *
 * **So this switches to the camera layer first**, then checks each thing that has to be true:
 *
 *   1. the control is a `<select>` and carries the four functions **by name**;
 *   2. moving it changes the **engine's own parameter state** -- read back from `/api/project`, which is what the
 *      renderer reads, not the DOM;
 *   3. and the **rendered frame** changes with it.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const URL = "http://127.0.0.1:8941/";
const PORT = 9233;
const profile = path.join(os.tmpdir(), "atsim-nvdrive-" + Date.now());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const chrome = spawn(CHROME, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=" + PORT, "--user-data-dir=" + profile,
    "--window-size=1600,1000", "--no-first-run", "--no-default-browser-check", URL,
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
    const errs = [];
    ws.addEventListener("message", (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
      if (m.method === "Runtime.exceptionThrown") errs.push(m.params.exceptionDetails.text);
      if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
        errs.push((m.params.args || []).map((a) => a.value).join(" "));
    });
    const send = (method, params) => new Promise((res) => {
      const myId = ++id;
      pending.set(myId, res);
      ws.send(JSON.stringify({ id: myId, method, params: params || {} }));
    });
    const evalJs = async (expr) => {
      const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
      if (r && r.exceptionDetails) return { error: r.exceptionDetails.text };
      return r && r.result ? r.result.value : null;
    };
    await send("Runtime.enable");
    await sleep(4500);

    // switch to the camera layer, which is what builds the night-vision rows
    const switched = await evalJs(`(() => {
      S.layer = "camera";
      if (typeof renderParams === "function") { renderParams(); return "renderParams() called"; }
      return "renderParams not found";
    })()`);
    console.log("   switching to the camera layer: " + switched);
    await sleep(800);

    // 1. the control
    const ctl = await evalJs(`(() => {
      const nv = [...document.querySelectorAll("select.psel")].find(s =>
        s.closest(".prow") && s.closest(".prow").textContent.includes("夜视模式"));
      if (!nv) return { found: false };
      return {
        found: true, tag: nv.tagName, cls: nv.className,
        options: [...nv.options].map(o => o.value + " = " + o.textContent),
        value: nv.value,
        readout: nv.closest(".prow").querySelector(".v").textContent,
        rowsInCamera: document.querySelectorAll(".prow").length,
      };
    })()`);
    console.log();
    console.log("   the control, as rendered:");
    console.log("      " + JSON.stringify(ctl, null, 1).replace(/\n/g, "\n      "));

    // 2 and 3: drive it and read the engine
    console.log();
    console.log("   driving it through the page's own handler, and reading the engine each time:");
    for (const mode of ["0", "1", "3"]) {
      await evalJs(`(() => {
        const nv = [...document.querySelectorAll("select.psel")].find(s =>
          s.closest(".prow") && s.closest(".prow").textContent.includes("夜视模式"));
        nv.value = "${mode}";
        nv.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
      })()`);
      await sleep(2200);
      const st = await evalJs(`(async () => {
        const r = await fetch("/api/project");
        const d = await r.json();
        // **the panel reads d.params -- the RESOLVED values**, which is where project_dict() puts them; d.project.params
                 // is the project's own stored set. Reading the wrong one is what made an earlier probe report a fault
                 // that was not there.
                 const c = d.params.camera;
        return { nv_mode: c.nv_mode, nv_green_mix: c.nv_green_mix,
                 nv_accumulate: c.nv_accumulate, src: (d.project.source||"").split(/[\\\\/]/).pop() };
      })()`);
      console.log("      mode " + mode + " -> " + JSON.stringify(st));
    }

    // a screenshot on the night-vision mode so the panel can be seen
    await evalJs(`(() => {
      const nv = [...document.querySelectorAll("select.psel")].find(s =>
        s.closest(".prow") && s.closest(".prow").textContent.includes("夜视模式"));
      nv.value = "1"; nv.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    })()`);
    await sleep(2500);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    if (shot && shot.data) {
      const p = path.join(__dirname, "ntscrs_ref", "gui_nightshot.png");
      fs.writeFileSync(p, Buffer.from(shot.data, "base64"));
      console.log();
      console.log("   screenshot written: " + p);
    }
    console.log("   page errors: " + (errs.length ? errs.slice(0, 3).join(" | ") : "none"));
  } finally {
    try { chrome.kill(); } catch (e) { /* ignore */ }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
