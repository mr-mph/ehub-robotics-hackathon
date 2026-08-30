"""Browser HUD: FastAPI + uvicorn in a background thread.

HUD.update(overlay_rgb, wrist_rgb, state) is called from the main loop; frames are JPEG-encoded once here and
served to all MJPEG clients. `state` is a plain dict, e.g. {ee_pose, holding, last_call, say, rules, voice_queue,
step, latency_ms}; GET /state is the single status payload (state sources registered with add_state_source() are
merged in at request time, e.g. state["calibration"]).

Actions: hud.register(name, fn, label, group) -> POST /action/{name} with a JSON body forwarded as kwargs, returning
{ok, message, data}; GET /actions lists them so the page renders one button row per group ("run", "robot",
"calibration", "voice"). label=None registers an action without a button (e.g. calib_sample, driven by clicking
the overhead image). Groups with nothing registered show "not available in this mode".
"""
from __future__ import annotations

import inspect
import json
import threading
import time
from typing import Callable, Iterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
GROUPS = ["run", "robot", "perception", "calibration", "voice", "rules", "models", "log"]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>sortbot HUD</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px/1.4 -apple-system,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:8px 12px;background:#181818;border-bottom:1px solid #333;position:sticky;top:0;z-index:5}
h1{font-size:14px;margin:0;color:#8ab;letter-spacing:2px}
.pillgrp{display:flex;align-items:center;gap:3px}
.pillgrp .lbl{color:#789;font-size:11px;text-transform:uppercase;margin-right:2px}
.pill{background:#222;border:1px solid #444;color:#aaa;border-radius:12px;padding:2px 10px;cursor:pointer;font:12px inherit}
.pill.sel{background:#264;border-color:#4a6;color:#dfd}
.pill.on{border-color:#3d3;box-shadow:0 0 5px #2a2}
#phase{font:13px Menlo,monospace;color:#fc8;text-transform:uppercase}
#stepc{font:13px Menlo,monospace;color:#8ab}
#estop{margin-left:auto;background:#a11;color:#fff;border:2px solid #f44;border-radius:6px;font-weight:700;font-size:15px;padding:8px 20px;cursor:pointer;letter-spacing:1px}
#estop:hover{background:#c22}
#estop.off{background:#f33;animation:blink 1s infinite}
@keyframes blink{50%{opacity:.45}}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:10px;padding:10px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:8px}
.card h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;color:#8ab}
img{width:100%;display:block;background:#000}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.3 Menlo,monospace;color:#cfc}
ul{margin:0;padding-left:18px}
.kv span{color:#8ab}
.side{display:flex;flex-direction:column;gap:10px}
.wrap{position:relative;cursor:crosshair}
#ring{position:absolute;border:2px solid #0f0;border-radius:50%;pointer-events:none;display:none;box-sizing:border-box}
button{background:#263;color:#dfd;border:1px solid #4a6;border-radius:4px;padding:4px 10px;margin:2px 4px 2px 0;cursor:pointer;font:13px inherit}
button:hover{background:#385}
.na{color:#666;font-style:italic}
.msg{color:#fc8;font:12px/1.3 Menlo,monospace;white-space:pre-wrap;margin-top:4px;min-height:14px}
.err{color:#f66;font:12px/1.3 Menlo,monospace;white-space:pre-wrap}
.note{color:#888;font-size:12px;margin-top:4px}
.tabs{display:flex;gap:2px;border-bottom:1px solid #333;margin-bottom:8px}
.tab{background:none;border:none;color:#89a;padding:4px 12px;cursor:pointer;border-bottom:2px solid transparent;border-radius:0;text-transform:uppercase;font-size:12px}
.tab.sel{color:#dfd;border-bottom-color:#4a6}
.tab:hover{background:#222}
.act{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:4px 0}
.act button{min-width:110px;margin:0}
input{background:#0d0d0d;color:#dfd;border:1px solid #444;border-radius:3px;padding:4px 6px;font:12px Menlo,monospace;width:72px}
input.wide{width:260px}
#ptt{background:#333;border:1px solid #666;color:#ddd;min-width:170px;padding:8px 12px}
#ptt.rec{background:#a11;border-color:#f44;animation:blink 1s infinite}
#transcript{color:#cfc;font:12px/1.3 Menlo,monospace;white-space:pre-wrap;margin-top:4px;min-height:14px}
.rl li{margin:2px 0}
.rl a{color:#8ab;cursor:pointer;margin-left:6px;text-decoration:none;border:1px solid #444;border-radius:3px;padding:0 5px;font-size:11px}
.rl a:hover{background:#333;color:#dfd}
select{background:#0d0d0d;color:#dfd;border:1px solid #444;border-radius:3px;padding:4px 6px;font:12px Menlo,monospace;max-width:280px}
.mrow{display:flex;align-items:center;gap:8px;margin:4px 0;flex-wrap:wrap}
.mrow .lbl2{color:#789;font-size:11px;text-transform:uppercase;width:80px}
#vlmstat{color:#fc8;font:12px Menlo,monospace}
.prow{display:flex;align-items:center;gap:8px;margin:4px 0}
.prow .lbl2{color:#789;font-size:11px;text-transform:uppercase;width:70px}
.prow input[type=range]{width:150px;accent-color:#4a6}
.prow .pv{font:12px Menlo,monospace;color:#cfc;width:56px}
.zrow{display:flex;align-items:center;gap:8px;margin:3px 0;font:12px Menlo,monospace;color:#cfc}
.zrow .zn{color:#fc8;width:84px}
.zrow button{min-width:0}
.zrow button.arm{background:#a60;border-color:#fc8}
#logbox{max-height:460px;overflow-y:auto}
.logrow{display:flex;gap:8px;margin:0 0 6px;border-bottom:1px solid #262626;padding-bottom:6px}
.logrow img{width:160px;height:auto;flex:none;background:#000;align-self:flex-start}
.logrow .lr{font:11px/1.35 Menlo,monospace;color:#cfc;white-space:pre-wrap;word-break:break-word;min-width:0}
.logrow.bad .lr{color:#f66}
.logrow .lt{color:#789}
[hidden]{display:none!important}
</style></head><body>
<header>
 <h1>SORTBOT</h1>
 <span class="pillgrp"><span class="lbl">robot</span><button class="pill" data-k="robot" data-v="mock">mock</button><button class="pill" data-k="robot" data-v="real">real</button><button class="pill" data-k="robot" data-v="off">off</button></span>
 <span class="pillgrp"><span class="lbl">cams</span><button class="pill" data-k="cams" data-v="sim">sim</button><button class="pill" data-k="cams" data-v="real">real</button><button class="pill" data-k="cams" data-v="off">off</button></span>
 <span class="pillgrp"><span class="lbl">vlm</span><button class="pill" data-k="vlm" data-v="mock">mock</button><button class="pill" data-k="vlm" data-v="live">live</button><button class="pill" data-k="vlm" data-v="off">off</button></span>
 <span id="phase">idle</span><span id="stepc"></span>
 <button id="estop" title="torque_off: cut motor torque and pause the run">E-STOP</button>
</header>
<div class="grid">
<div class="side">
 <div class="card"><h3>Overhead (VLM input) &mdash; click the calibration target to pick its colour</h3>
  <div class="wrap"><img id="ov" src="/overhead.mjpg"><div id="ring"></div></div></div>
 <div class="card"><h3>Wrist</h3><img id="wr" src="/wrist.mjpg"></div>
</div>
<div class="side">
 <div class="card"><h3>Status</h3><div class="kv" id="kv"></div><div class="err" id="errline"></div><div class="msg" id="msg"></div></div>
 <div class="card"><div class="tabs" id="tabs"></div><div id="acts"></div>
  <div data-extra="calibration"><div class="kv" id="calib"></div><div class="msg" id="calibmsg"></div>
   <div class="note">Hold the target in the gripper, Start, move the arm with the leader, Capture at &gt;= 4 spread-out spots
   (Touch table once with the fingertip on the table), Finish. Recalibrate whenever the overhead camera or the robot base moves,
   or the overlay grid stops lining up with the table.</div></div>
  <div data-extra="voice">
   <div class="act"><button id="ptt" title="Hold to record; on release the clip is transcribed (ElevenLabs) and fed to the bot like a spoken correction.">&#127908; Hold to talk</button></div>
   <div class="note" id="micnote"></div><div id="transcript"></div>
   <h3 style="margin-top:8px">Queue</h3><ul id="voice"></ul></div>
  <div data-extra="rules"><h3 style="margin-top:8px">Rules (persisted; sent with every VLM prompt)</h3><ul id="rlist" class="rl"></ul>
   <h3 style="margin-top:8px">Hints (one-shot, this run)</h3><ul id="hlist"></ul></div>
  <div data-extra="models"><div id="mrows"><span class="na">loading...</span></div>
   <div class="note">last VLM call: <span id="vlmstat">-</span></div><div class="note" id="mnotes"></div></div>
  <div data-extra="perception"><div id="prows"><span class="na">loading...</span></div>
   <div class="note">foreground if V &gt; v_min or S &gt; s_min; blobs kept if area_min &lt;= px area &lt;= area_max.
   Changes apply live and persist to config.yaml. Toggle mask streams the binary mask on the overhead view.</div>
   <h3 style="margin-top:8px">Zones</h3><div id="zrows"></div>
   <div class="note">"set drop" then click the overhead image at the new drop point (converted px&rarr;mm, persisted).</div></div>
  <div data-extra="log"><div id="logbox"><span class="na">no log entries yet</span></div></div>
 </div>
 <div class="card"><h3>Last VLM call</h3><pre id="call"></pre><pre id="say" style="color:#fc8"></pre></div>
 <div class="card"><h3>Rules</h3><ul id="rules"></ul></div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const li=(id,a)=>{$(id).innerHTML=(a||[]).map(x=>'<li>'+esc(x)+'</li>').join('')||'<li style="color:#666">none</li>'};
const GROUPS=['run','robot','perception','calibration','voice','rules','models','log'];
const HDR=new Set(['set_mode','torque_off']);
let actions=[],actsig='',curTab='run',ring=null,frameW=640,frameH=480;
async function act(name,body){try{
 const r=await fetch('/action/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
 const j=await r.json();const m=(j.ok?'':'!! ')+(j.message||'');
 $('msg').textContent=m;if(name.startsWith('calib_'))$('calibmsg').textContent=m;
 loadActions();return j;}catch(e){$('msg').textContent='!! '+e;return{ok:false};}}
function groupList(){const gs=[...GROUPS];for(const a of actions)if(!gs.includes(a.group))gs.push(a.group);return gs;}
function renderTabs(){const gs=groupList();if(!gs.includes(curTab))curTab=gs[0];
 $('tabs').innerHTML=gs.map(g=>`<button class="tab${g===curTab?' sel':''}" data-g="${esc(g)}">${esc(g)}</button>`).join('');
 $('tabs').querySelectorAll('.tab').forEach(b=>b.onclick=()=>{curTab=b.dataset.g;renderTabs();});
 renderBody();}
function rowHtml(a){const ins=(a.params||[]).map(p=>{const d=p.default==null?'':String(p.default);
 const w=(p.name==='text'||p.name==='task')?' class="wide"':'';
 return `<input${w} data-p="${esc(p.name)}" placeholder="${esc(p.name)}" value="${esc(d)}">`;}).join('');
 return `<div class="act"><button data-n="${esc(a.name)}" title="${esc(a.help||'')}">${esc(a.label)}</button>${ins}</div>`;}
function renderBody(){const rows=actions.filter(a=>a.group===curTab&&a.label&&!HDR.has(a.name));
 $('acts').innerHTML=rows.length?rows.map(rowHtml).join(''):'<span class="na">not available in this mode</span>';
 $('acts').querySelectorAll('button[data-n]').forEach(b=>b.onclick=()=>{const body={};
  b.parentElement.querySelectorAll('input').forEach(i=>{const v=i.value.trim();if(v==='')return;
   body[i.dataset.p]=/^-?\\d+(\\.\\d+)?$/.test(v)?parseFloat(v):v;});
  act(b.dataset.n,body);});
 document.querySelectorAll('[data-extra]').forEach(d=>d.hidden=d.dataset.extra!==curTab);
 if(curTab==='models'&&!window._models)loadModels();
 if(curTab==='log'){logSig='';loadLog();}}
async function loadActions(){try{const a=await(await fetch('/actions')).json();
 const sig=JSON.stringify(a);if(sig!==actsig){actsig=sig;actions=a;renderTabs();}}catch(e){}}
document.querySelectorAll('.pill').forEach(b=>b.onclick=()=>act('set_mode',{[b.dataset.k]:b.dataset.v}));
$('estop').onclick=()=>act('torque_off');
$('ov').onclick=async e=>{const im=$('ov'),r=im.getBoundingClientRect();
 const nw=im.naturalWidth||frameW,nh=im.naturalHeight||frameH;
 const u=(e.clientX-r.left)/r.width*nw,v=(e.clientY-r.top)/r.height*nh;
 if(dropZone){const z=dropZone;dropZone=null;zoneSig='';
  const j=await act('px_to_mm',{u:Math.round(u),v:Math.round(v)});
  if(j.ok&&j.data)await act('set_zone_drop',{name:z,x:j.data.x,y:j.data.y});
  return;}
 const j=await act('calib_sample',{u:Math.round(u),v:Math.round(v)});
 ring=j.data&&j.data.det?j.data.det:null;drawRing();};
function drawRing(){const im=$('ov'),el=$('ring');if(!ring){el.style.display='none';return;}
 const nw=im.naturalWidth||frameW,nh=im.naturalHeight||frameH,sx=im.clientWidth/nw,sy=im.clientHeight/nh,[u,v,rad]=ring;
 el.style.display='block';el.style.left=(u-rad)*sx+'px';el.style.top=(v-rad)*sy+'px';el.style.width=2*rad*sx+'px';el.style.height=2*rad*sy+'px';}
async function tick(){try{
 const s=await (await fetch('/state')).json();
 const r=s.run||{},rb=s.robot,p=(rb&&rb.ee_pose)||s.ee_pose||{};
 $('phase').textContent=r.phase||'idle';
 $('stepc').textContent='step '+(r.step??s.step??0)+'/'+(r.max_steps??'-');
 document.querySelectorAll('.pill').forEach(b=>{const cur=(r.mode||{})[b.dataset.k];
  b.classList.toggle('sel',cur===b.dataset.v);
  b.classList.toggle('on',cur===b.dataset.v&&!!((r.connected||{})[b.dataset.k]));});
 const toff=rb&&rb.torque===false;
 $('estop').classList.toggle('off',!!toff);
 $('estop').textContent=toff?'TORQUE OFF':'E-STOP';
 $('errline').textContent=r.last_error?('last error: '+r.last_error):'';
 const jd=rb&&rb.joints_deg?rb.joints_deg.map(x=>x.toFixed(0)).join(' '):null;
 $('kv').innerHTML=
  `<div><span>status</span> ${esc(s.status??'-')} &nbsp; <span>latency</span> ${s.latency_ms??'-'} ms &nbsp; <span>result</span> ${esc(r.result||'-')}</div>`+
  `<div><span>task</span> ${esc(r.task||'(default: sort sensibly)')}</div>`+
  `<div><span>EE</span> x=${fmt(p.x)} y=${fmt(p.y)} z=${fmt(p.z)} roll=${fmt(p.roll_deg)}`+(jd?` &nbsp; <span>joints</span> ${jd}`:'')+`</div>`+
  `<div><span>holding</span> ${(rb?rb.holding:s.holding)??'nothing'} &nbsp; <span>gripper</span> ${grip(rb?rb.gripper_open:s.gripper_open)} &nbsp; <span>torque</span> ${rb?(rb.torque?'on':'OFF'):'-'}</div>`+
  `<div><span>updated</span> ${s.age_s??'-'} s ago</div>`;
 if(s.frame_wh){frameW=s.frame_wh[0];frameH=s.frame_wh[1];}
 const c=s.calibration;
 if(c&&c.state){const fk=c.fk_mm?c.fk_mm.map(x=>x.toFixed(0)).join(', '):'-';const t=c.target||{};
  $('calib').innerHTML=`<div><span>state</span> ${esc(c.state)} &nbsp; <span>samples</span> ${c.n??0}`+
   (c.residual_mean_mm!=null?` &nbsp; <span>residual</span> mean ${c.residual_mean_mm} max ${c.residual_max_mm} mm`:'')+`</div>`+
   `<div><span>FK</span> ${fk} mm &nbsp; <span>z offset</span> ${c.z_offset_mm==null?'-':c.z_offset_mm.toFixed(1)} mm</div>`+
   `<div><span>target</span> ${esc(t.name||'-')} hsv ${esc(JSON.stringify(t.hsv_lo||[]))}..${esc(JSON.stringify(t.hsv_hi||[]))} &nbsp; <span>detected</span> ${c.det?c.det.map(x=>x.toFixed(0)).join(','):'no'}</div>`;
  if(c.message)$('calibmsg').textContent=c.message;
  if(c.state==='running'){ring=c.det||null;drawRing();}}
 else $('calib').innerHTML='<span class="na">not available (connect a robot)</span>';
 $('call').textContent=s.last_call?JSON.stringify(s.last_call):'-';
 $('say').textContent=s.say?'"'+s.say+'"':'';
 const rl=s.rules;
 li('rules',Array.isArray(rl)?rl:(rl?(rl.list||[]).concat((rl.hints||[]).map(h=>'(hint) '+h)):[]));
 renderRules(rl);
 li('voice',(s.voice&&s.voice.queue)||s.voice_queue);
 if(s.voice&&s.voice.last_transcript&&!$('transcript').textContent.startsWith('!!')&&$('transcript').textContent!=='transcribing...')
  $('transcript').textContent='heard: "'+s.voice.last_transcript+'"';
 const vs=s.vlm;
 const vtxt=vs?(vs.model+'  '+(vs.last_latency_ms==null?'- ms':vs.last_latency_ms+' ms')
  +(vs.last_cost_usd!=null?'  ~$'+vs.last_cost_usd.toFixed(4)+'/call':'')
  +(vs.last_usage?'  ('+vs.last_usage.input_tokens+' in / '+vs.last_usage.output_tokens+' out)':'')):'-';
 document.querySelectorAll('#vlmstat,.vlmbeside').forEach(el=>el.textContent=vtxt);
 renderPerc(s.perception);
}catch(e){}}
const fmt=v=>v==null?'-':(typeof v==='number'?v.toFixed(0):v);
const grip=g=>g===undefined||g===null?'-':(g?'open':'closed');
let rec=null,chunks=[];
function b64(buf){const u=new Uint8Array(buf);let t='';for(let i=0;i<u.length;i+=0x8000)t+=String.fromCharCode.apply(null,u.subarray(i,i+0x8000));return btoa(t);}
async function pttDown(){if(rec)return;
 if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia||typeof MediaRecorder==='undefined'){
  $('micnote').textContent='microphone unavailable in this browser context (getUserMedia/MediaRecorder missing; use Send to bot instead)';return;}
 try{const st=await navigator.mediaDevices.getUserMedia({audio:true});
  rec=new MediaRecorder(st,MediaRecorder.isTypeSupported&&MediaRecorder.isTypeSupported('audio/webm')?{mimeType:'audio/webm'}:undefined);
  chunks=[];rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  rec.onstop=async()=>{st.getTracks().forEach(t=>t.stop());const b=new Blob(chunks,{type:rec.mimeType||'audio/webm'});rec=null;pttIdle();
   if(b.size<200){$('micnote').textContent='clip too short, try again';return;}
   $('transcript').textContent='transcribing...';
   const j=await act('transcribe',{audio_b64:b64(await b.arrayBuffer()),mime:b.type||'audio/webm'});
   $('transcript').textContent=j.ok&&j.data&&j.data.text?('heard: "'+j.data.text+'"'):('!! '+(j.message||'transcription failed'));};
  rec.start();$('ptt').classList.add('rec');$('ptt').textContent='recording... release to send';$('micnote').textContent='';}
 catch(e){$('micnote').textContent='mic unavailable: '+e+' (use Send to bot instead)';rec=null;pttIdle();}}
function pttIdle(){$('ptt').classList.remove('rec');$('ptt').innerHTML='&#127908; Hold to talk';}
function pttUp(){if(rec&&rec.state!=='inactive')rec.stop();else pttIdle();}
$('ptt').onmousedown=pttDown;$('ptt').onmouseup=pttUp;$('ptt').onmouseleave=pttUp;
$('ptt').ontouchstart=e=>{e.preventDefault();pttDown();};$('ptt').ontouchend=e=>{e.preventDefault();pttUp();};
let rulesSig='';
function renderRules(rl){if(!rl||Array.isArray(rl)||!$('rlist'))return;const sig=JSON.stringify(rl);if(sig===rulesSig)return;rulesSig=sig;
 $('rlist').innerHTML=(rl.list||[]).map((r,i)=>`<li>${esc(r)}<a data-mv="${i}|up" title="move up (higher priority)">&#8593;</a><a data-mv="${i}|down" title="move down">&#8595;</a><a data-del="${i}" title="delete this rule">&#10005;</a></li>`).join('')||'<li style="color:#666">none</li>';
 $('rlist').querySelectorAll('a[data-mv]').forEach(a=>a.onclick=()=>{const[i,d]=a.dataset.mv.split('|');rulesSig='';act('move_rule',{i:+i,dir:d});});
 $('rlist').querySelectorAll('a[data-del]').forEach(a=>a.onclick=()=>{rulesSig='';act('delete_rule',{i:+a.dataset.del});});
 $('hlist').innerHTML=(rl.hints||[]).map(h=>'<li>'+esc(h)+'</li>').join('')||'<li style="color:#666">none</li>';}
const PROVIDERS=[['openai','openai model'],['elevenlabs_tts','tts model'],['elevenlabs_stt','stt model'],['elevenlabs_voice','voice']];
async function loadModels(){const j=await act('get_models');if(j.ok&&j.data){window._models=j.data;renderModels();}
 else $('mrows').innerHTML='<span class="na">'+esc(j.message||'get_models failed')+'</span>';}
function renderModels(){const d=window._models;if(!d)return;const cur=d.current||{};
 const opts={openai:d.openai||[],elevenlabs_tts:(d.elevenlabs||{}).tts||[],elevenlabs_stt:(d.elevenlabs||{}).stt||[],
  elevenlabs_voice:((d.elevenlabs||{}).voices||[]).map(v=>[v.id,v.name+' ('+v.id.slice(0,8)+')'])};
 $('mrows').innerHTML=PROVIDERS.map(([p,lbl])=>{let os=opts[p].map(o=>{const[v,n]=Array.isArray(o)?o:[o,o];
   return `<option value="${esc(v)}"${v===cur[p]?' selected':''}>${esc(n)}</option>`;}).join('');
  if(cur[p]&&!opts[p].some(o=>(Array.isArray(o)?o[0]:o)===cur[p]))os=`<option value="${esc(cur[p])}" selected>${esc(cur[p])}</option>`+os;
  return `<div class="mrow"><span class="lbl2">${esc(lbl)}</span><select data-prov="${esc(p)}" title="Hot-swaps the live client and persists to config.yaml.">${os}</select>`+
   (p==='openai'?' <span class="vlmbeside" style="color:#fc8;font:12px Menlo,monospace"></span>':'')+`</div>`;}).join('');
 $('mnotes').textContent=(d.notes||[]).join('; ');
 $('mrows').querySelectorAll('select').forEach(sl=>sl.onchange=async()=>{await act('set_model',{provider:sl.dataset.prov,value:sl.value});window._models=null;loadModels();});}
// ---- perception tab: threshold sliders + zone drop points ----
const SL=[['v_min',0,255,1],['s_min',0,255,1],['area_min',0,5000,10],['area_max',1000,120000,500]];
let percBuilt=false,zoneSig='',dropZone=null,dragging=null;
function buildPerc(){percBuilt=true;
 $('prows').innerHTML=SL.map(([n,lo,hi,st])=>`<div class="prow"><span class="lbl2">${n}</span><input type="range" data-s="${n}" min="${lo}" max="${hi}" step="${st}" title="Applies live on release; persisted to config.yaml perception:"><span class="pv" data-v="${n}">-</span></div>`).join('');
 $('prows').querySelectorAll('input[type=range]').forEach(sl=>{
  sl.oninput=()=>{dragging=sl.dataset.s;$('prows').querySelector(`[data-v="${sl.dataset.s}"]`).textContent=sl.value;};
  sl.onchange=async()=>{dragging=null;await act('set_detector_params',{[sl.dataset.s]:parseFloat(sl.value)});act('redetect');};});}
function renderPerc(pc){if(!pc||!pc.params)return;if(!percBuilt)buildPerc();
 for(const[n]of SL){const sl=$('prows').querySelector(`input[data-s="${n}"]`);
  if(sl&&dragging!==n){sl.value=pc.params[n];$('prows').querySelector(`[data-v="${n}"]`).textContent=pc.params[n];}}
 const sig=JSON.stringify([pc.zones,dropZone]);if(sig===zoneSig)return;zoneSig=sig;
 $('zrows').innerHTML=(pc.zones||[]).map(z=>`<div class="zrow"><span class="zn">${esc(z.name)}</span><span>drop (${z.drop[0]}, ${z.drop[1]})</span><button data-z="${esc(z.name)}" class="${dropZone===z.name?'arm':''}" title="Click, then click the overhead image at the new drop point.">${dropZone===z.name?'click overhead...':'set drop'}</button></div>`).join('');
 $('zrows').querySelectorAll('button[data-z]').forEach(b=>b.onclick=()=>{dropZone=dropZone===b.dataset.z?null:b.dataset.z;zoneSig='';});}
// ---- log tab: decision ring buffer, newest first ----
let logSig='';
async function loadLog(){if(curTab!=='log')return;try{
 const l=await(await fetch('/log')).json();const sig=l.length?l[0].i+'/'+l.length:'0';if(sig===logSig)return;logSig=sig;
 $('logbox').innerHTML=l.map(e=>`<div class="logrow${e.ok===false?' bad':''}">${e.thumb_b64?`<img src="data:image/jpeg;base64,${e.thumb_b64}">`:''}`+
  `<div class="lr"><span class="lt">#${e.step} ${new Date(e.t*1000).toLocaleTimeString()}${e.latency_ms!=null?' '+e.latency_ms+'ms':''}</span>\n${esc(e.tool)}(${esc(JSON.stringify(e.args||{}))})\n${esc(e.result||'')}${e.say?'\\n"'+esc(e.say)+'"':''}</div></div>`).join('')
  ||'<span class="na">no log entries yet</span>';}catch(e){}}
setInterval(loadLog,1500);
loadActions();setInterval(tick,500);tick();setInterval(loadActions,5000);
</script></body></html>"""


def _encode(rgb: np.ndarray | None) -> bytes | None:
    if rgb is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _jsonable(v):
    if hasattr(v, "__dataclass_fields__"):
        return {k: _jsonable(getattr(v, k)) for k in v.__dataclass_fields__}
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class HUD:
    def __init__(self, port: int = 8765, host: str = "127.0.0.1"):
        self.port, self.host = port, host
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frames: dict[str, bytes | None] = {"overhead": None, "wrist": None}
        self._seq = 0
        self._state: dict = {}
        self._frame_wh: tuple[int, int] | None = None
        self._t_update = 0.0
        self._actions: dict[str, dict] = {}
        self._sources: dict[str, Callable[[], dict]] = {}
        self.app = FastAPI()
        self.app.get("/", response_class=HTMLResponse)(lambda: PAGE)
        self.app.get("/state")(self._get_state)
        self.app.get("/actions")(self._get_actions)
        self.app.post("/action/{name}")(self._post_action)
        self.app.get("/overhead.mjpg")(lambda: self._mjpeg("overhead"))
        self.app.get("/wrist.mjpg")(lambda: self._mjpeg("wrist"))
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    # --- main-loop API ---
    def update(self, overlay_frame: np.ndarray | None, wrist_frame: np.ndarray | None, state: dict) -> None:
        ov, wr = _encode(overlay_frame), _encode(wrist_frame)
        with self._cond:
            if ov is not None:
                self._frames["overhead"] = ov
                self._frame_wh = (overlay_frame.shape[1], overlay_frame.shape[0])
            if wr is not None:
                self._frames["wrist"] = wr
            if state:
                self._state = _jsonable(state)
            self._t_update = time.time()
            self._seq += 1
            self._cond.notify_all()

    def register(self, name: str, fn: Callable[..., object], label: str | None = None, group: str = "run",
                 params: list[dict] | None = None, help: str | None = None) -> None:
        """Expose fn as POST /action/{name}; JSON body -> kwargs. label=None: no button on the page.
        params (auto-derived from fn's signature if omitted) tells the page which input fields to render:
        [{"name": ..., "default": ...}]. help: one sentence (what this does / when to use it), served in
        GET /actions for tooltips."""
        if params is None:
            try:
                params = [{"name": q.name, "default": None if q.default is inspect.Parameter.empty else q.default}
                          for q in inspect.signature(fn).parameters.values()
                          if q.kind in (q.POSITIONAL_OR_KEYWORD, q.KEYWORD_ONLY)]
            except (TypeError, ValueError):
                params = []
        self._actions[name] = {"name": name, "label": label, "group": group, "params": params, "help": help, "fn": fn}

    def add_state_source(self, key: str, fn: Callable[[], dict]) -> None:
        """fn() is merged into GET /state under `key` at request time."""
        self._sources[key] = fn

    def add_route(self, path: str, fn: Callable[[], object]) -> None:
        """Serve fn() as JSON at GET `path` (e.g. /log) -- for payloads too heavy to ride along in /state."""
        self.app.get(path)(lambda: JSONResponse(_jsonable(fn())))

    def start(self) -> None:
        cfg = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="hud")
        self._thread.start()
        t0 = time.time()
        while not self._server.started and time.time() - t0 < 5:
            time.sleep(0.05)
        print(f"[hud] http://{self.host}:{self.port}/")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=5)

    # --- handlers ---
    def _get_state(self):
        with self._lock:
            s = dict(self._state)
            s["age_s"] = round(time.time() - self._t_update, 1) if self._t_update else None
            s["seq"], s["frame_wh"] = self._seq, self._frame_wh
        for k, fn in self._sources.items():
            try:
                s[k] = _jsonable(fn())
            except Exception as e:  # noqa: BLE001
                s[k] = {"state": "error", "message": str(e)}
        return JSONResponse(s)

    def _get_actions(self):
        return JSONResponse([{k: a[k] for k in ("name", "label", "group", "params", "help")} for a in self._actions.values()])

    async def _post_action(self, name: str, request: Request):
        a = self._actions.get(name)
        if a is None:
            return JSONResponse({"ok": False, "message": f"action {name!r} not available in this mode", "data": None}, status_code=404)
        try:
            body = await request.body()
            kw = json.loads(body) if body else {}
            r = a["fn"](**kw)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "message": f"{name}: {e}", "data": None})
        if isinstance(r, dict) and "ok" in r:
            out = {"ok": bool(r["ok"]), "message": str(r.get("message", "")), "data": r.get("data")}
        elif isinstance(r, bool) or r is None:
            out = {"ok": r is not False, "message": "", "data": None}
        else:
            out = {"ok": True, "message": str(r) if isinstance(r, str) else "", "data": None if isinstance(r, str) else r}
        return JSONResponse(_jsonable(out))

    def _mjpeg(self, name: str) -> StreamingResponse:
        def gen() -> Iterator[bytes]:
            last = -1
            while not (self._server and self._server.should_exit):
                with self._cond:
                    self._cond.wait_for(lambda: self._seq != last or (self._server and self._server.should_exit), timeout=1.0)
                    last, jpg = self._seq, self._frames[name]
                if jpg is None:
                    continue
                yield _BOUNDARY % len(jpg) + jpg + b"\r\n"
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def _selftest() -> None:
    import socket
    import urllib.error
    import urllib.request

    from sortbot.calibration import ColorTarget, detect_target
    from sortbot.types import Pose

    with socket.socket() as sk:  # free port: avoid clashing with a running main loop / demo
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    hud = HUD(port=port)
    frame = np.full((480, 640, 3), 30, np.uint8)
    cv2.circle(frame, (420, 300), 18, (40, 220, 60), -1)  # synthetic green target
    calib = {"state": "idle", "n": 0}

    def sample(u, v):
        t = ColorTarget.from_sample(frame, u, v)
        det = detect_target(frame, t)
        return {"ok": det is not None, "message": "sampled", "data": {"target": t.to_dict(), "det": det}}

    hud.register("calib_sample", sample, None, "calibration")
    hud.register("calib_start", lambda: calib.update(state="running") or "started", "Start calibration", "calibration",
                 help="Begin a teleop calibration session.")
    hud.add_state_source("calibration", lambda: calib)
    hud.start()
    try:
        for i in range(2):
            ov = np.zeros((360, 640, 3), np.uint8); ov[:, :, i] = 200
            wr = np.full((240, 320, 3), 60 * (i + 1), np.uint8)
            hud.update(ov, wr, {"ee_pose": Pose(150, 0, 180), "holding": None, "step": i,
                                "latency_ms": 1234, "last_call": {"tool": "pick", "args": {"id": 3}},
                                "say": "picking the red wire", "rules": ["wires go left"],
                                "voice_queue": ["put sensors in the middle"]})
        base = f"http://127.0.0.1:{port}"
        post = lambda path, body: json.load(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
            base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"), timeout=3))
        with urllib.request.urlopen(base + "/", timeout=3) as r:
            assert r.status == 200 and b"overhead.mjpg" in r.read()
        with urllib.request.urlopen(base + "/state", timeout=3) as r:
            s = json.load(r)
            assert s["step"] == 1 and s["ee_pose"]["x"] == 150 and s["frame_wh"] == [640, 360]
            assert s["calibration"] == {"state": "idle", "n": 0}
        with urllib.request.urlopen(base + "/actions", timeout=3) as r:
            acts = json.load(r)
            assert {a["name"] for a in acts} == {"calib_sample", "calib_start"} and all(a["group"] == "calibration" for a in acts)
            byname = {a["name"]: a for a in acts}
            assert byname["calib_start"]["help"] == "Begin a teleop calibration session." and byname["calib_sample"]["help"] is None
        j = post("/action/calib_sample", {"u": 420, "v": 300})
        assert j["ok"] and abs(j["data"]["det"][0] - 420) < 1 and abs(j["data"]["det"][1] - 300) < 1, j
        assert 40 <= j["data"]["target"]["hsv_lo"][0] <= 70, j
        j = post("/action/calib_sample", {"u": 10, "v": 10})  # background: nothing that colour is a blob... except everything
        assert "target" in j["data"], j
        j = post("/action/calib_start", {})
        assert j["ok"] and j["message"] == "started", j
        with urllib.request.urlopen(base + "/state", timeout=3) as r:
            assert json.load(r)["calibration"]["state"] == "running"
        try:
            post("/action/nope", {})
            raise AssertionError("unknown action accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 404 and "not available" in json.load(e)["message"]
        with urllib.request.urlopen(base + "/overhead.mjpg", timeout=3) as r:
            assert r.status == 200 and "multipart/x-mixed-replace" in r.headers["Content-Type"], r.headers
            assert r.read(64).startswith(b"--frame")
        with urllib.request.urlopen(base + "/wrist.mjpg", timeout=3) as r:
            assert r.read(16).startswith(b"--frame")
    finally:
        hud.stop()
    print("selftest OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        from sortbot import config; h = HUD(port=config.load().hud_port)
        h.start(); print("Ctrl-C to stop")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            h.stop()
