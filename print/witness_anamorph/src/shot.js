const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch({args:['--use-gl=swiftshader','--enable-unsafe-swiftshader']});
const shots=JSON.parse(process.argv[2]);
for(const s of shots){const p=await b.newPage({viewport:{width:s.w||800,height:s.h||800}});
 const u='http://localhost:8765/render.html?'+new URLSearchParams({f:JSON.stringify(s.f),cam:JSON.stringify(s.cam),fov:s.fov||40,w:s.w||800,h:s.h||800,dir:s.dir||'out',basic:s.basic||''});
 await p.goto(u);await p.waitForFunction('window.DONE',null,{timeout:120000});await p.screenshot({path:s.out});await p.close();}
await b.close();})();
