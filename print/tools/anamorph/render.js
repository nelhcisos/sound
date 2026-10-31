// node render.js scene.json cams.json outdir   (THREE=/path/three.module.js)
const fs = require('fs'), path = require('path');
const { chromium } = require('playwright');
(async () => {
  const [sceneF, camsF, outdir] = process.argv.slice(2);
  const three = process.env.THREE;
  const tpl = fs.readFileSync(path.join(__dirname, 'render.html'), 'utf8').replace('THREE_PATH', 'file://' + three);
  const tmp = path.join(path.dirname(path.resolve(sceneF)), '_render.html');
  fs.writeFileSync(tmp, tpl);
  const b = await chromium.launch({args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--allow-file-access-from-files']});
  const p = await b.newPage();
  p.on('pageerror', e => console.log('pageerror', e.message));
  await p.goto('file://' + tmp); await p.waitForFunction('window.ready === true', {timeout: 30000});
  const S = JSON.parse(fs.readFileSync(sceneF, 'utf8')), cams = JSON.parse(fs.readFileSync(camsF, 'utf8'));
  fs.mkdirSync(outdir, {recursive: true});
  const urls = await p.evaluate(([S, cams]) => window.renderAll(S, cams), [S, cams]);
  urls.forEach((u, i) => fs.writeFileSync(path.join(outdir, cams[i].name + '.png'), Buffer.from(u.split(',')[1], 'base64')));
  await b.close(); fs.unlinkSync(tmp); console.log('rendered', urls.length);
})();
