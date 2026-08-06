#!/usr/bin/env python3
"""Static experiment dashboard: scans experiments/<name>/train/*/ and writes
one self-contained report.html (no server, no CDN — open from the filesystem).

Per run it reads summary.csv + args.yaml (always present) and run.json +
coco_metrics.json (written by run_metrics.py; older runs may lack them).
Regenerated automatically after every training run; run manually any time:

    python3 build_report.py [--output-dir experiments/falcon-vision-effdet]
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml


def collect(output_dir, session=None):
    """Session layout: <output>/<session-dt>/<level>/train/summary.csv."""
    runs = []
    pattern = f"{session}/*/train/summary.csv" if session else "*/*/train/summary.csv"
    for sc in sorted(output_dir.glob(pattern)):
        d = sc.parent                     # …/<level>/train
        lvl = d.parent                    # …/<level>
        args_y = yaml.safe_load((d / "args.yaml").read_text()) if (d / "args.yaml").exists() else {}
        rows = [{k: float(v) if k != "epoch" else int(v) for k, v in r.items()}
                for r in csv.DictReader(open(sc))]
        manifest = None
        for mf in (d / "run.json", lvl / "run.json"):
            if mf.exists():
                manifest = json.loads(mf.read_text())
                break
        run = {
            "name": f"{lvl.parent.name}/{lvl.name}",
            "model": args_y.get("model", "?"),
            "summary": rows,
            "manifest": manifest,
            "metrics": json.loads((d / "coco_metrics.json").read_text()) if (d / "coco_metrics.json").exists() else None,
        }
        runs.append(run)
    return runs


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Falcon Vision OD — experiments</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7;
  --ink-1:#0b0b0b; --ink-2:#52514e; --ink-mut:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d;
    --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-mut:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-mut:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
.viz-root { margin:0; background:var(--page); color:var(--ink-1);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; padding:24px; }
h1 { font-size:20px; margin:0 0 2px; } h2 { font-size:15px; margin:0 0 10px; }
.sub { color:var(--ink-2); margin-bottom:20px; }
.card { background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:16px 18px; margin-bottom:18px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
@media (max-width:900px){ .grid2{grid-template-columns:1fr} }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid); }
th:first-child,td:first-child { text-align:left; }
th { color:var(--ink-2); font-weight:600; cursor:pointer; user-select:none; white-space:nowrap; }
tbody tr { cursor:pointer; }
tbody tr:hover { background:color-mix(in srgb, var(--ink-1) 4%, transparent); }
tbody tr.sel { background:color-mix(in srgb, var(--s1) 12%, transparent); }
.best { font-weight:700; }
svg text { fill:var(--ink-mut); font:11px system-ui,sans-serif; }
svg .axis { stroke:var(--axis); stroke-width:1; }
svg .grid { stroke:var(--grid); stroke-width:1; }
svg .lbl { fill:var(--ink-2); font-weight:600; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin:4px 0 8px; color:var(--ink-2); font-size:12px; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.key { width:14px; height:2px; display:inline-block; border-radius:1px; }
.tip { position:absolute; pointer-events:none; background:var(--surface-1);
  border:1px solid var(--border); border-radius:8px; padding:8px 10px;
  font-size:12px; box-shadow:0 2px 10px rgba(0,0,0,.12); display:none; z-index:5;
  font-variant-numeric:tabular-nums; }
.tip b { color:var(--ink-1); } .tip .r { display:flex; gap:8px; align-items:center; justify-content:space-between; }
.chartwrap { position:relative; }
details { margin-top:10px; color:var(--ink-2); } summary { cursor:pointer; }
.mut { color:var(--ink-mut); } code { font-size:12px; color:var(--ink-2); }
</style></head>
<body class="viz-root">
<h1>Falcon Vision OD — experiments</h1>
<div class="sub">__NRUNS__ runs · generated __WHEN__ · headline = car AP (94% of objects); 6-class mean is a footnote</div>

<div class="card">
<h2>Runs <span class="mut">(click a row to inspect · click headers to sort)</span></h2>
<table id="runs"><thead><tr>
<th data-k="name">run</th><th data-k="model">model</th><th data-k="label_source">labels</th>
<th data-k="train_images">train imgs</th><th data-k="epochs_run">epochs</th>
<th data-k="car_ap">car AP</th><th data-k="car_ap_large">car AP-lg</th>
<th data-k="person_ap">person AP</th><th data-k="best_eval_map">mean mAP</th>
</tr></thead><tbody></tbody></table>
</div>

<div class="grid2">
<div class="card"><h2 id="t-loss">Loss</h2><div class="legend" id="lg-loss"></div>
  <div class="chartwrap" id="c-loss"></div></div>
<div class="card"><h2 id="t-map">Val mAP (vs run's own val labels)</h2>
  <div class="chartwrap" id="c-map"></div></div>
</div>
<div class="grid2">
<div class="card"><h2>Per-class AP <span class="mut" id="t-cls"></span></h2>
  <div class="chartwrap" id="c-cls"></div></div>
<div class="card"><h2>Val mAP across runs <span class="mut">(each vs its OWN val labels —
  runs with different label sources are not directly comparable)</span></h2>
  <div class="legend" id="lg-all"></div>
  <div class="chartwrap" id="c-all"></div></div>
</div>

<div class="card"><details><summary>Per-epoch table (selected run)</summary>
<table id="epochs"><thead><tr><th>epoch</th><th>train loss</th><th>eval loss</th><th>eval mAP</th></tr></thead>
<tbody></tbody></table></details>
<div class="mut" id="rundir"></div></div>

<script>
const DATA = __DATA__;
const S = n => getComputedStyle(document.body).getPropertyValue('--s'+n).trim();
const fmt = (v,d=3) => v==null||isNaN(v) ? '—' : (+v).toFixed(d);
let sel = DATA.length-1;

/* ---------- tiny svg helpers ---------- */
function mk(tag, attrs, parent){ const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; }
function niceTicks(max){ if(!(max>0)) return [0,1]; const raw=max/4,
  p=Math.pow(10,Math.floor(Math.log10(raw))), c=[1,2,2.5,5,10].find(c=>c*p>=raw)*p,
  t=[]; for(let v=0; v<=max+1e-9; v+=c) t.push(+v.toFixed(6)); return t; }

/* line chart: series=[{name,color,pts:[[x,y],...]}] */
function linechart(el, series, {endLabel=false}={}){
  el.innerHTML=''; const W=el.clientWidth||520, H=240, L=44, R=endLabel?58:14, T=10, B=24;
  const svg=mk('svg',{width:'100%',viewBox:`0 0 ${W} ${H}`,role:'img'},el);
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])), ys=series.flatMap(s=>s.pts.map(p=>p[1]));
  const xmax=Math.max(...xs), xmin=Math.min(...xs), ymax=Math.max(...ys)*1.05||1;
  const X=x=>L+(x-xmin)/(xmax-xmin||1)*(W-L-R), Y=y=>H-B-(y/ymax)*(H-T-B);
  for(const t of niceTicks(ymax)){ mk('line',{x1:L,x2:W-R,y1:Y(t),y2:Y(t),class:'grid'},svg);
    mk('text',{x:L-6,y:Y(t)+3,'text-anchor':'end'},svg).textContent=+t.toFixed(3); }
  mk('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'},svg);
  const xticks=[...new Set(niceTicks(xmax).map(Math.round))].filter(t=>t>=xmin&&t<=xmax);
  for(const t of xticks) mk('text',{x:X(t),y:H-B+15,'text-anchor':'middle'},svg).textContent=t;
  for(const s of series){
    const dpath=s.pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('');
    mk('path',{d:dpath,fill:'none',stroke:s.color,'stroke-width':2,
      'stroke-linejoin':'round','stroke-linecap':'round'},svg);
    const last=s.pts[s.pts.length-1];
    mk('circle',{cx:X(last[0]),cy:Y(last[1]),r:4,fill:s.color,
      stroke:'var(--surface-1)','stroke-width':2},svg);
    if(endLabel) mk('text',{x:X(last[0])+8,y:Y(last[1])+4,class:'lbl'},svg)
      .textContent=fmt(last[1]);
  }
  /* crosshair + tooltip (one tooltip, every series) */
  const tip=document.createElement('div'); tip.className='tip'; el.appendChild(tip);
  const hair=mk('line',{y1:T,y2:H-B,class:'axis',visibility:'hidden'},svg);
  svg.addEventListener('pointermove',ev=>{
    const r=svg.getBoundingClientRect(), px=(ev.clientX-r.left)*(W/r.width);
    const x=Math.round(xmin+(px-L)/(W-L-R)*(xmax-xmin));
    if(x<xmin||x>xmax){tip.style.display='none';hair.setAttribute('visibility','hidden');return;}
    hair.setAttribute('x1',X(x)); hair.setAttribute('x2',X(x)); hair.setAttribute('visibility','visible');
    tip.innerHTML=''; const h=document.createElement('div'); h.append('epoch '+x); h.className='mut'; tip.appendChild(h);
    for(const s of series){ const p=s.pts.find(p=>p[0]===x); if(!p) continue;
      const row=document.createElement('div'); row.className='r';
      const k=document.createElement('span'); const sw=document.createElement('i');
      sw.className='key'; sw.style.background=s.color; k.append(sw,' '+s.name);
      const v=document.createElement('b'); v.textContent=fmt(p[1],4);
      row.append(k,v); tip.appendChild(row); }
    tip.style.display='block';
    const ex=ev.clientX-r.left, flip=ex>r.width*0.6;
    tip.style.left=flip?'':(ex+14)+'px'; tip.style.right=flip?(r.width-ex+14)+'px':'';
    tip.style.top='18px';
  });
  svg.addEventListener('pointerleave',()=>{tip.style.display='none';hair.setAttribute('visibility','hidden');});
}

/* horizontal bars: items=[{name,value,note}] — single series, labels at the tip */
function barchart(el, items){
  el.innerHTML=''; const W=el.clientWidth||520, ROW=34, L=86, R=60,
  H=items.length*ROW+8, max=Math.max(...items.map(i=>i.value||0),0.01)*1.08;
  const svg=mk('svg',{width:'100%',viewBox:`0 0 ${W} ${H}`,role:'img'},el);
  const tip=document.createElement('div'); tip.className='tip'; el.appendChild(tip);
  items.forEach((it,i)=>{
    const y=i*ROW+6, bh=Math.min(22,ROW-12), w=Math.max(2,(it.value||0)/max*(W-L-R));
    mk('text',{x:L-8,y:y+bh/2+4,'text-anchor':'end',class:'lbl'},svg).textContent=it.name;
    const bar=mk('path',{d:`M${L} ${y} h${Math.max(0,w-4)} a4 4 0 0 1 4 4 v${bh-8} a4 4 0 0 1 -4 4 h-${Math.max(0,w-4)} z`,
      fill:'var(--s1)'},svg);
    mk('text',{x:L+w+8,y:y+bh/2+4,class:'lbl'},svg).textContent=fmt(it.value);
    const hit=mk('rect',{x:0,y:y-3,width:W,height:ROW-2,fill:'transparent'},svg);
    hit.addEventListener('pointermove',ev=>{ const r=svg.getBoundingClientRect();
      tip.innerHTML=''; const b=document.createElement('b'); b.textContent=fmt(it.value);
      tip.append(b,' '+it.name+(it.note?' · '+it.note:'')); tip.style.display='block';
      tip.style.left=(ev.clientX-r.left+12)+'px'; tip.style.top=(y*(r.height/H)-30)+'px'; });
    hit.addEventListener('pointerleave',()=>tip.style.display='none');
    bar.style.pointerEvents='none';
  });
}

/* ---------- table + wiring ---------- */
function rowvals(r){ const m=r.manifest||{}, best=r.summary.reduce((a,b)=>b.eval_map>a.eval_map?b:a);
  return { name:r.name, model:r.model, label_source:m.label_source||'—',
    train_images:m.train_images??null, epochs_run:r.summary.length,
    car_ap:m.car_ap??null, car_ap_large:m.car_ap_large??null,
    person_ap:m.person_ap??null, best_eval_map:best.eval_map }; }

let sortK='name', sortAsc=true;
function renderTable(){
  const tb=document.querySelector('#runs tbody'); tb.innerHTML='';
  const rows=DATA.map((r,i)=>({i,v:rowvals(r)}));
  rows.sort((a,b)=>{ const x=a.v[sortK], y=b.v[sortK];
    return (x==null)-(y==null) || (x>y?1:x<y?-1:0)*(sortAsc?1:-1); });
  const bests={}; for(const k of ['car_ap','car_ap_large','person_ap','best_eval_map'])
    bests[k]=Math.max(...rows.map(r=>r.v[k]??-1));
  for(const {i,v} of rows){ const tr=document.createElement('tr');
    if(i===sel) tr.className='sel';
    for(const k of ['name','model','label_source','train_images','epochs_run',
                    'car_ap','car_ap_large','person_ap','best_eval_map']){
      const td=document.createElement('td');
      td.textContent = typeof v[k]==='number' && k!=='train_images' && k!=='epochs_run'
        ? fmt(v[k]) : (v[k]??'—');
      if(typeof v[k]==='number' && v[k]===bests[k]) td.classList.add('best');
      tr.appendChild(td); }
    tr.addEventListener('click',()=>{ sel=i; render(); });
    tb.appendChild(tr); }
}
document.querySelectorAll('#runs th').forEach(th=>th.addEventListener('click',()=>{
  const k=th.dataset.k; if(sortK===k) sortAsc=!sortAsc; else {sortK=k;sortAsc=true;} renderTable(); }));

function legend(el, entries){ el.innerHTML='';
  for(const e of entries){ const s=document.createElement('span');
    const k=document.createElement('i'); k.className='key'; k.style.background=e.color;
    s.append(k, ' '+e.name); el.appendChild(s); } }

function render(){
  renderTable();
  const r=DATA[sel];
  document.getElementById('t-loss').textContent='Loss — '+r.name;
  const loss=[{name:'train loss',color:S(1),pts:r.summary.map(e=>[e.epoch,e.train_loss])},
              {name:'eval loss',color:S(2),pts:r.summary.map(e=>[e.epoch,e.eval_loss])}];
  legend(document.getElementById('lg-loss'), loss);
  linechart(document.getElementById('c-loss'), loss);
  linechart(document.getElementById('c-map'),
    [{name:'eval mAP',color:S(1),pts:r.summary.map(e=>[e.epoch,e.eval_map])}],{endLabel:true});
  const cls=document.getElementById('c-cls'), note=document.getElementById('t-cls');
  if(r.metrics){ note.textContent='';
    barchart(cls, Object.entries(r.metrics.per_class)
      .filter(([,v])=>v.gt_boxes>0).sort((a,b)=>b[1].ap-a[1].ap)
      .map(([n,v])=>({name:n,value:v.ap,note:v.gt_boxes+' gt boxes, AP-large '+fmt(v.ap_large)}))); }
  else { cls.innerHTML='<div class="mut">no coco_metrics.json — backfill: python3 run_metrics.py &lt;run dir&gt;</div>'; note.textContent=''; }
  const shown=DATA.slice(-8);
  const all=shown.map((x,j)=>({name:x.name,color:S(j+1),
    pts:x.summary.map(e=>[e.epoch,e.eval_map])}));
  legend(document.getElementById('lg-all'), all);
  linechart(document.getElementById('c-all'), all);
  const et=document.querySelector('#epochs tbody'); et.innerHTML='';
  for(const e of r.summary){ const tr=document.createElement('tr');
    for(const v of [e.epoch,fmt(e.train_loss,4),fmt(e.eval_loss,4),fmt(e.eval_map,4)]){
      const td=document.createElement('td'); td.textContent=v; tr.appendChild(td); }
    et.appendChild(tr); }
  document.getElementById('rundir').textContent='run dir: '+(r.manifest?.run_dir||('experiments/…/train/'+r.name));
}
render();
addEventListener('resize',()=>render());
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("experiments/falcon-vision-effdet"))
    ap.add_argument("--session", default=None,
                    help="restrict to one session dir and write its own "
                         "<session>/report.html")
    a = ap.parse_args()
    runs = collect(a.output_dir, a.session)
    if not runs:
        raise SystemExit(f"no runs with summary.csv under {a.output_dir}")
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(runs))
            .replace("__NRUNS__", str(len(runs)))
            .replace("__WHEN__", datetime.now().strftime("%Y-%m-%d %H:%M")))
    out = (a.output_dir / a.session / "report.html") if a.session \
        else (a.output_dir / "report.html")
    out.write_text(html)
    print(f"[report] {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
