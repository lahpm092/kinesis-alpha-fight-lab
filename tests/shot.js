/* Visual QA: serve the lab, open every route in headless Chrome, screenshot.
   Real waits only - the store loads over HTTP and three.js needs frames.
   Usage: node tests/shot.js  (writes work/shots/*.png) */
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const PCORE = "/Users/hive/Claude Code/kinesis-alpha-decision-lab/web/node_modules/puppeteer-core";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 5199;

const puppeteer = require(PCORE);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const outDir = path.join(ROOT, "work", "shots");
  fs.mkdirSync(outDir, { recursive: true });
  const srv = spawn("python3", [path.join(ROOT, "lab", "server.py"), "--port", String(PORT)], { stdio: "ignore" });
  await sleep(1200);
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: "new",
    args: ["--no-first-run", "--disable-gpu-sandbox", "--window-size=1560,1100"],
  });
  const errors = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1560, height: 1100, deviceScaleFactor: 2 });
    page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
    page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });

    const shots = [
      ["bout", "#/bout", 6000],
      ["bout-mid", "#/bout?t=95", 6000],
      ["analysis", "#/analysis", 4000],
      ["method", "#/method", 2500],
    ];
    for (const [name, hash, wait] of shots) {
      await page.goto(`http://127.0.0.1:${PORT}/${hash}`, { waitUntil: "networkidle2", timeout: 60000 });
      await page.reload({ waitUntil: "networkidle2", timeout: 60000 });
      await sleep(wait);
      await page.screenshot({ path: path.join(outDir, name + ".png"), fullPage: name !== "bout" });
      console.log("shot", name);
    }
    // fight view interactions: toggle body mode + angles, mid-bout
    await page.goto(`http://127.0.0.1:${PORT}/#/bout?t=95`, { waitUntil: "networkidle2", timeout: 60000 });
    await page.reload({ waitUntil: "networkidle2", timeout: 60000 });
    await sleep(5200);
    const btns = await page.$$(".fv-hud .fbtn");
    for (const b of btns.slice(0, 2)) await b.click();       // both to body mode
    const labels = await page.$$eval(".fv-hud .fbtn", (els) => els.map((e) => e.textContent));
    const angleIdx = labels.findIndex((t) => t === "angles");
    if (angleIdx >= 0) await (await page.$$(".fv-hud .fbtn"))[angleIdx].click();
    await sleep(1200);
    const fv = await page.$(".fightview");
    await fv.screenshot({ path: path.join(outDir, "fightview-body-angles.png") });
    console.log("shot fightview-body-angles");
  } finally {
    await browser.close();
    srv.kill();
  }
  const real = errors.filter((e) => !/favicon|ERR_ABORTED/.test(e));
  if (real.length) {
    console.log("ERRORS:");
    for (const e of real) console.log("  " + e);
    process.exit(1);
  }
  console.log("no page errors");
})();
