// node views.js <dir> <prefix> '<json list of {name, ...view}>'
const { chromium } = require('playwright');
(async () => {
  const [dir, prefix, list] = process.argv.slice(2);
  const views = JSON.parse(list);
  const W = 1600, H = 800;
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: W, height: H } });
  p.on('pageerror', e => console.log('err:', e.message));
  p.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  await p.goto(`http://127.0.0.1:8765/render.html?dir=${dir}&w=${W}&h=${H}`);
  await p.waitForFunction(() => window.DONE, null, { timeout: 120000 });
  for (const v of views) {
    await p.evaluate(v => window.setView(v), v);
    await p.screenshot({ path: `${prefix}${v.name}.png` });
    console.log('saved', v.name);
  }
  await b.close();
})();
