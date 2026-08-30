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

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>sortbot HUD</title>
<style>
:root{
 --bg:#0d1117;--bg1:#151b23;--bg2:#1c2430;--line:#28313d;--line2:#39465a;
 --tx:#dde6f0;--tx1:#93a1b3;--tx2:#5f6b7a;
 --acc:#37b183;--acc2:#26845f;--red:#e5484d;--amber:#e8a33d;
 --mono:ui-monospace,Menlo,Consolas,monospace;
 --r:10px;--rs:6px;--sh:0 4px 14px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.45 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
[hidden]{display:none!important}
header{display:flex;align-items:center;gap:12px;padding:8px 14px;background:var(--bg1);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50;min-height:56px}
h1{font-size:15px;margin:0;letter-spacing:3px;color:var(--acc);font-weight:700}
.conns{display:flex;gap:10px;cursor:pointer;flex:none}
.conn{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--tx1);text-transform:uppercase;letter-spacing:.5px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--tx2);flex:none}
.conn.on .dot{background:var(--acc);box-shadow:0 0 6px var(--acc)}
.conn .cv{color:var(--tx2);text-transform:none}
.conn.on .cv{color:var(--tx)}
#banner{flex:1;min-width:120px;font-size:14px;font-weight:600;padding:8px 12px;border-radius:var(--rs);background:var(--bg2);border:1px solid var(--line);border-left:4px solid var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer;min-height:38px;line-height:20px}
#banner.ok{border-left-color:var(--acc)}
#banner.amber{border-left-color:var(--amber)}
#banner.err{border-left-color:var(--red);color:#ffb3b5}
#micchip{color:#fff;background:var(--red);border-radius:20px;padding:8px 14px;font-weight:700;font-size:12px;animation:blink 1.2s infinite;flex:none}
.btn{background:var(--bg2);color:var(--tx);border:1px solid var(--line2);border-radius:var(--rs);padding:9px 14px;min-height:40px;cursor:pointer;font:inherit;font-weight:500}
.btn:hover:not(:disabled){background:#243040;border-color:#4a5a72}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.primary{background:var(--acc2);border-color:var(--acc);color:#eafff5}
.btn.primary:hover:not(:disabled){background:var(--acc)}
.btn.danger{background:#5a2427;border-color:var(--red);color:#ffd9da}
.btn.danger:hover:not(:disabled){background:#7a2d31}
.btn.big{min-height:46px;padding:10px 18px;font-size:14px;font-weight:600}
.btn.sm{min-height:32px;padding:4px 10px;font-size:12px}
.btn.busy::after{content:'';display:inline-block;width:11px;height:11px;border:2px solid transparent;border-top-color:currentColor;border-radius:50%;margin-left:7px;vertical-align:-2px;animation:spin .7s linear infinite}
.btn[data-stat=ok]::after{content:'\2713';margin-left:7px;color:var(--acc)}
.btn.primary[data-stat=ok]::after{color:#fff}
.btn[data-stat=err]::after{content:'\2715';margin-left:7px;color:var(--red)}
#estop{background:#8c1d20;color:#fff;border:2px solid var(--red);border-radius:var(--rs);font-weight:800;font-size:15px;letter-spacing:1px;padding:10px 22px;min-height:46px;cursor:pointer;flex:none}
#estop:hover{background:#b02428}
#estop.off{background:var(--red);animation:blink 1s infinite}
@keyframes blink{50%{opacity:.5}}
@keyframes spin{to{transform:rotate(360deg)}}
#checklist{display:flex;align-items:center;gap:6px;margin:10px 14px 0;padding:8px 14px;background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);flex-wrap:wrap}
#checklist .ttl{font-weight:700;font-size:11px;color:var(--tx1);margin-right:4px;text-transform:uppercase;letter-spacing:1.5px}
#checklist .st{display:flex;align-items:center;gap:8px;color:var(--tx1);text-decoration:none;padding:6px 10px;border-radius:var(--rs);min-height:36px}
#checklist .st:hover{background:var(--bg2)}
#checklist .st .n{width:22px;height:22px;border-radius:50%;background:var(--bg2);border:1px solid var(--line2);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex:none}
#checklist .st.cur{background:var(--bg2);color:var(--tx);outline:1px solid var(--acc)}
#checklist .st.cur .n{background:var(--acc);border-color:var(--acc);color:#04120c}
#checklist .st.done{color:var(--acc)}
#checklist .st.done .n{background:transparent;border-color:var(--acc);color:var(--acc)}
#checklist .sep{color:var(--tx2)}
#ckx{margin-left:auto;background:none;border:none;color:var(--tx2);cursor:pointer;font-size:16px;padding:6px 12px;min-height:36px}
#ckx:hover{color:var(--tx)}
.layout{display:grid;grid-template-columns:minmax(0,65fr) minmax(0,35fr);gap:12px;padding:12px 14px;align-items:start}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
.viewwrap{position:relative;background:#000;border:1px solid var(--line);border-radius:var(--r);overflow:hidden;cursor:crosshair}
#ov{width:100%;display:block;min-height:200px;background:#000}
#ring{position:absolute;border:2px solid #22ff88;border-radius:50%;pointer-events:none;display:none;box-sizing:border-box}
#imgage{position:absolute;left:8px;top:8px;font:11px var(--mono);background:rgba(0,0,0,.55);color:#9fe8c9;padding:3px 8px;border-radius:4px;pointer-events:none;z-index:5}
#imgage.warn{color:#ffd9a0}
#imgage.stale{color:#ffb3b5;background:rgba(90,20,22,.7)}
.pip{position:absolute;right:10px;bottom:10px;width:26%;min-width:120px;border:1px solid var(--line2);border-radius:var(--rs);overflow:hidden;background:#000;box-shadow:var(--sh);cursor:default}
.pip img{width:100%;display:block;min-height:40px}
.pip span{position:absolute;left:6px;bottom:4px;font-size:10px;color:#cfd8e3;text-shadow:0 1px 2px #000;text-transform:uppercase;letter-spacing:1px}
#readout{margin-top:10px;padding:8px 12px;background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);font:12px/1.7 var(--mono);color:var(--tx1);min-height:60px;white-space:pre-wrap;word-break:break-word}
#readout b{color:var(--tx);font-weight:600}
.panel{background:var(--bg1);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.tabbar{display:flex;border-bottom:1px solid var(--line);background:var(--bg2)}
.tabbar .tb{flex:1;background:none;border:none;color:var(--tx1);padding:12px 6px;min-height:44px;font:inherit;font-weight:600;font-size:13px;cursor:pointer;border-bottom:2px solid transparent}
.tabbar .tb:hover{color:var(--tx)}
.tabbar .tb.sel{color:var(--acc);border-bottom-color:var(--acc);background:var(--bg1)}
#helpbtn{flex:0 0 44px;background:none;border:none;border-left:1px solid var(--line);color:var(--tx1);font-size:15px;font-weight:700;cursor:pointer;min-height:44px}
#helpbtn.sel{color:var(--amber)}
.tabsec{padding:4px 14px 14px}
.sec{padding:12px 0;border-bottom:1px solid var(--line)}
.sec:last-child{border-bottom:none}
.sec h3{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--tx1)}
.inline{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
.btnrow{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
input,select{background:#0b0f14;color:var(--tx);border:1px solid var(--line2);border-radius:var(--rs);padding:9px 10px;min-height:40px;font:12px var(--mono);width:80px}
input:focus,select:focus{outline:1px solid var(--acc);border-color:var(--acc)}
.grow{flex:1;min-width:140px;width:auto}
select{width:auto;max-width:100%}
.help{display:none;color:var(--tx2);font-size:11.5px;margin:2px 0 4px;line-height:1.4;flex-basis:100%}
body.showhelp .help{display:block}
.rowmsg{min-height:16px;font:11.5px var(--mono);color:var(--amber);white-space:pre-wrap;word-break:break-word;margin-top:4px}
.rowmsg.err{color:#ff8a8e}
.why{min-height:15px;font-size:11.5px;color:var(--tx2);font-style:italic;margin-top:3px}
.note{color:var(--tx2);font-size:11.5px;margin-top:6px;line-height:1.5}
.mono{font:12px/1.5 var(--mono);color:var(--tx1);white-space:pre-wrap;word-break:break-word}
.empty{color:var(--tx2);font-style:italic;font-size:12px}
.crow{display:flex;align-items:center;gap:10px;margin:8px 0;flex-wrap:wrap}
.crow .cn{font-weight:700;font-size:13px;width:110px}
.crow .cd{color:var(--tx1);font-size:11.5px;flex:1;min-width:120px}
.crow .cst{font:11px var(--mono);color:var(--tx2);width:78px;text-align:right}
.crow .cst.on{color:var(--acc)}
.guide{margin:8px 0;padding:0;list-style:none;counter-reset:g}
.guide li{position:relative;padding:7px 8px 7px 36px;color:var(--tx1);border-radius:var(--rs);counter-increment:g;font-size:12.5px;line-height:1.45;margin:2px 0}
.guide li::before{content:counter(g);position:absolute;left:7px;top:8px;width:20px;height:20px;border-radius:50%;background:var(--bg2);border:1px solid var(--line2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--tx1)}
.guide li.cur{background:var(--bg2);color:var(--tx);outline:1px solid var(--acc)}
.guide li.cur::before{background:var(--acc);border-color:var(--acc);color:#04120c;content:counter(g)}
.guide li.done{color:var(--acc)}
.guide li.done::before{content:'\2713';background:transparent;border-color:var(--acc);color:var(--acc)}
.zrow{display:flex;align-items:center;gap:8px;margin:6px 0;font:12px var(--mono);color:var(--tx1);flex-wrap:wrap}
.zrow .zn{color:var(--amber);width:82px;font-weight:600}
.zrow .btn.arm{background:#4d3607;border-color:var(--amber);color:#ffe2ac}
.mrow{display:flex;align-items:center;gap:8px;margin:8px 0;flex-wrap:wrap}
.mrow .lbl2{color:var(--tx1);font-size:11px;text-transform:uppercase;width:70px;flex:none}
#vlmstat{color:var(--amber);font:11.5px var(--mono)}
.rl{margin:4px 0;padding:0;list-style:none}
.rl li{display:flex;align-items:center;gap:6px;padding:7px 8px;margin:4px 0;background:var(--bg2);border:1px solid var(--line);border-radius:var(--rs);font-size:12.5px}
.rl li .rt{flex:1;min-width:0;word-break:break-word}
.rl li a{color:var(--tx1);cursor:pointer;border:1px solid var(--line2);border-radius:4px;padding:4px 8px;min-width:28px;text-align:center;text-decoration:none;font-size:12px}
.rl li a:hover{background:var(--bg1);color:var(--tx)}
#ptt.rec{background:var(--red);border-color:var(--red);animation:blink 1s infinite;color:#fff}
#mictoggle.live{background:var(--red);border-color:var(--red);color:#fff;font-weight:700}
#transcript{font:12px var(--mono);color:#bfe8d6;min-height:18px;white-space:pre-wrap;margin-top:4px}
.jog{display:grid;grid-template-columns:repeat(4,minmax(56px,1fr));gap:6px;margin:6px 0}
#logbox{max-height:420px;overflow-y:auto;margin-top:6px}
.logrow{display:flex;gap:10px;margin:0 0 8px;border-bottom:1px solid var(--line);padding-bottom:8px}
.logrow img{width:150px;height:auto;flex:none;background:#000;border-radius:4px;align-self:flex-start}
.logrow .lr{font:11px/1.45 var(--mono);color:var(--tx1);white-space:pre-wrap;word-break:break-word;min-width:0}
.logrow.bad .lr{color:#ff9b9e}
.logrow .lt{color:var(--tx2)}
.arow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:6px 0}
.arow .an{font:11px var(--mono);color:var(--tx2);flex-basis:100%}
#toasts{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:8px;z-index:300;max-width:360px}
.toast{background:var(--bg2);border:1px solid var(--line2);border-left:4px solid var(--acc);color:var(--tx);padding:10px 14px;border-radius:var(--rs);box-shadow:var(--sh);font-size:12.5px;animation:rise .25s ease-out;word-break:break-word}
.toast.err{border-left-color:var(--red)}
.toast.say{border-left-color:var(--amber)}
@keyframes rise{from{opacity:0;transform:translateY(8px)}}
#demo{position:fixed;inset:0;background:var(--bg);z-index:200;display:flex;flex-direction:column;padding:18px;gap:10px}
#demo .drow{display:flex;align-items:center;gap:14px}
#d-banner{flex:1;font-size:22px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#demo .dview{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000;border-radius:var(--r);overflow:hidden}
#demo .dview img{max-width:100%;max-height:100%;object-fit:contain}
#d-decision{font:14px/1.5 var(--mono);color:var(--tx1);min-height:44px}
#d-say{color:var(--amber);font-size:18px;font-weight:600;min-height:26px}
.ticker{overflow:hidden;border-top:1px solid var(--line);padding-top:8px;white-space:nowrap}
.ticker .tk{display:inline-block;padding-left:100%;animation:tick 28s linear infinite;color:var(--tx1);font-size:14px}
@keyframes tick{to{transform:translateX(-100%)}}
</style></head><body>
<header>
 <h1>SORTBOT</h1>
 <div class="conns" id="conns" title="Device connections - click to open Setup">
  <span class="conn" id="cd-robot"><i class="dot"></i>robot <span class="cv">off</span></span>
  <span class="conn" id="cd-cams"><i class="dot"></i>cams <span class="cv">off</span></span>
  <span class="conn" id="cd-vlm"><i class="dot"></i>vlm <span class="cv">off</span></span>
 </div>
 <div id="banner" title="">starting...</div>
 <span id="micchip" hidden>&#9679; MIC LIVE</span>
 <button id="torqon" class="btn primary" hidden>Torque on</button>
 <button id="demobtn" class="btn" title="Fullscreen view for judging: stream + last decision + rules, no controls">Demo</button>
 <button id="estop" title="E-STOP (key: e) - cut motor torque and pause the run">E-STOP</button>
</header>
<div id="checklist" hidden>
 <span class="ttl">First run</span>
 <a class="st" id="ck1" href="#"><span class="n">1</span>Connect devices</a><span class="sep">&rarr;</span>
 <a class="st" id="ck2" href="#"><span class="n">2</span>Calibrate</a><span class="sep">&rarr;</span>
 <a class="st" id="ck3" href="#"><span class="n">3</span>Start sorting</a>
 <button id="ckx" title="Hide this checklist">&#10005;</button>
</div>
<main class="layout">
<section class="stage">
 <div class="viewwrap" id="ovwrap">
  <img id="ov" src="/overhead.mjpg" alt="overhead camera">
  <div id="imgage" title="Age of the overhead image on the server - if this climbs, the stream is stalled">no image yet</div>
  <div id="ring"></div>
  <div class="pip"><img id="wr" src="/wrist.mjpg" alt="wrist camera"><span>wrist</span></div>
 </div>
 <div id="readout">waiting for state...</div>
</section>
<aside class="panel">
 <nav class="tabbar">
  <button class="tb sel" data-tab="setup">Setup</button>
  <button class="tb" data-tab="operate">Operate</button>
  <button class="tb" data-tab="tune">Tune</button>
  <button class="tb" data-tab="debug">Debug</button>
  <button id="helpbtn" title="Show a one-line explanation under every control">?</button>
 </nav>

 <section class="tabsec" id="tab-setup">
  <div class="sec" id="sec-mode" hidden>
   <h3>1 &middot; Connect the devices</h3>
   <div id="connrows"></div>
   <div class="note">Cameras work without the robot (live preview, nothing moves). Sorting needs all three connected.</div>
   <div class="rowmsg" id="msg-setup"></div>
  </div>
  <div class="sec" id="sec-calib" hidden>
   <h3>2 &middot; Calibrate the overhead camera</h3>
   <div class="mono" id="calstat">-</div>
   <ol class="guide" id="calguide"></ol>
   <div class="btnrow" id="calbtns"></div>
   <div class="why" id="why-calib"></div>
   <div class="rowmsg" id="msg-calib"></div>
   <div class="note"><b>&#9888; Not a straight line!</b> Samples that lie on one line cannot be fitted &mdash; spread them across the whole mat in both directions.
   Recalibrate whenever the overhead camera or the robot base is moved or bumped, or the zone outlines stop lining up with the table.</div>
  </div>
  <div id="extra-setup"></div>
 </section>

 <section class="tabsec" id="tab-operate" hidden>
  <div class="sec" id="sec-task" hidden>
   <h3>Task</h3>
   <div class="inline"><input id="task" class="grow" placeholder="e.g. group similar items; big things go in LEFT"><button id="taskbtn" class="btn">Set task</button></div>
   <div class="help" data-h="set_task"></div>
   <div class="note">current: <span id="taskcur" class="mono">(default: sort sensibly)</span></div>
  </div>
  <div class="sec" id="sec-run" hidden>
   <h3>Run</h3>
   <div class="btnrow">
    <button id="b-start" class="btn primary big" data-h-title="start">&#9654; Start</button>
    <button id="b-pause" class="btn big" data-h-title="pause">&#10073;&#10073; Pause</button>
    <button id="b-resume" class="btn big" data-h-title="resume">&#9654; Resume</button>
    <button id="b-stop" class="btn danger big" data-h-title="stop">&#9632; Stop</button>
   </div>
   <div class="why" id="why-run"></div>
   <div class="inline"><button id="b-step" class="btn" data-h-title="step_once">Step once</button>
    <input id="maxsteps" placeholder="40" size="4"><button id="b-maxsteps" class="btn" data-h-title="set_max_steps">Set max steps</button></div>
   <div class="help" data-h="step_once"></div>
   <div class="rowmsg" id="msg-run"></div>
  </div>
  <div class="sec" id="sec-voice" hidden>
   <h3>Talk to the bot</h3>
   <div class="inline"><input id="corr" class="grow" placeholder='e.g. "shiny things go in the RIGHT zone"'><button id="corrbtn" class="btn">Send</button></div>
   <div class="help" data-h="say_to_bot"></div>
   <div class="inline"><button id="ptt" class="btn" data-h-title="transcribe">&#127908; Hold to talk</button>
    <button id="mictoggle" class="btn" hidden>Listening: OFF</button></div>
   <div class="help" data-h="mic_on"></div>
   <div class="note" id="micnote"></div>
   <div id="transcript"></div>
   <div class="note">queued for the next step: <span id="vqueue" class="mono">none</span></div>
   <div class="rowmsg" id="msg-voice"></div>
  </div>
  <div class="sec" id="sec-rules" hidden>
   <h3>Rules <span style="text-transform:none;letter-spacing:0">(sent with every prompt, kept across runs)</span></h3>
   <ul id="rlist" class="rl"></ul>
   <div class="inline"><input id="newrule" class="grow" placeholder="e.g. red things go in LEFT"><button id="rulebtn" class="btn" data-h-title="add_rule">Add rule</button></div>
   <div class="help" data-h="add_rule"></div>
   <div class="inline"><span class="note" style="margin:0">one-shot hints: <span id="hints" class="mono">none</span></span>
    <button id="b-clearhints" class="btn sm" data-h-title="clear_hints">Clear hints</button></div>
   <div class="rowmsg" id="msg-rules"></div>
  </div>
  <div id="extra-operate"></div>
 </section>

 <section class="tabsec" id="tab-tune" hidden>
  <div class="sec" id="sec-models" hidden>
   <h3>Models</h3>
   <div id="mrows"><span class="empty">loading model lists...</span></div>
   <div class="help" data-h="set_model"></div>
   <div class="note">last VLM call: <span id="vlmstat">-</span></div>
   <div class="note" id="mnotes"></div>
   <div class="rowmsg" id="msg-models"></div>
  </div>
  <div class="sec" id="sec-zones" hidden>
   <h3>Zone drop points</h3>
   <div id="zrows"><span class="empty">no zones</span></div>
   <div class="help" data-h="set_zone_drop"></div>
   <div class="note">Press a zone's <b>set drop</b>, then click the overhead image where that zone should drop objects. Saved to config.yaml.</div>
   <div class="rowmsg" id="msg-zones"></div>
  </div>
  <div id="extra-tune"></div>
 </section>

 <section class="tabsec" id="tab-debug" hidden>
  <div class="sec" id="sec-robot" hidden>
   <h3>Manual arm control</h3>
   <div class="btnrow">
    <button id="b-home" class="btn" data-h-title="home">Home</button>
    <button id="b-open" class="btn" data-h-title="open_gripper">Open gripper</button>
    <button id="b-close" class="btn" data-h-title="close_gripper">Close gripper</button>
    <button id="b-ton" class="btn" data-h-title="torque_on">Torque on</button>
   </div>
   <div class="inline"><span class="note" style="margin:0">jog step</span><input id="jogd" value="10" size="4"><span class="note" style="margin:0">mm (deg for roll)</span></div>
   <div class="jog" id="jogpad"></div>
   <div class="help" data-h="jog"></div>
   <div class="inline"><input id="gx" placeholder="x"><input id="gy" placeholder="y"><input id="gz" placeholder="z"><button id="b-goto" class="btn" data-h-title="goto">Go to</button></div>
   <div class="help" data-h="goto"></div>
   <div class="why" id="why-robot"></div>
   <div class="rowmsg" id="msg-robot"></div>
  </div>
  <div class="sec" id="sec-log" hidden>
   <h3>Decision log</h3>
   <div class="btnrow"><button id="b-logclear" class="btn sm" data-h-title="log_clear">Clear log</button></div>
   <div id="logbox"><span class="empty">no decisions yet &mdash; start a run in Operate</span></div>
  </div>
  <div class="sec">
   <details><summary style="cursor:pointer;color:var(--tx1)">All actions (raw API)</summary>
   <div class="note">Every registered action, straight from GET /actions. Buttons above are shortcuts to these same endpoints.</div>
   <div id="rawlist"></div></details>
  </div>
  <div id="extra-debug"></div>
 </section>
</aside>
</main>
<div id="toasts"></div>
<div id="demo" hidden>
 <div class="drow"><div id="d-banner">-</div>
  <button id="d-estop" class="btn danger big">E-STOP</button>
  <button id="d-close" class="btn big" title="Exit demo view (Esc)">&#10005; Exit</button></div>
 <div class="dview"><img id="d-ov" alt="overhead"></div>
 <div id="d-say"></div>
 <div id="d-decision">-</div>
 <div class="ticker"><span class="tk" id="d-ticker">rules: none yet</span></div>
</div>
<script>
'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const TABS=['setup','operate','tune','debug'];
const TABOF={run:'operate',robot:'debug',perception:'tune',calibration:'setup',voice:'operate',rules:'operate',models:'tune',log:'debug'};
const CLAIMED=new Set(['connect_robot','connect_cameras','connect_vlm','start','pause','resume','stop','step_once','set_max_steps','set_task',
 'home','open_gripper','close_gripper','jog','goto','torque_off','torque_on',
 'say_to_bot','say_to_robot','transcribe','speak','mic_on','mic_off',
 'add_rule','delete_rule','move_rule','clear_hints','get_models','set_model',
 'set_zone_drop','px_to_mm','log_clear',
 'calib_start','calib_touch','calib_capture','calib_undo','calib_finish','calib_cancel','calib_sample']);
const DEVICES=[
 ['robot','connect_robot','Robot','SO-101 follower arm. Torque comes on when connected; the arm moves only on your actions.'],
 ['cams','connect_cameras','Cameras','Overhead + wrist. Works without the robot for a live preview.'],
 ['vlm','connect_vlm','Vision model','The OpenAI planner (needs OPENAI_API_KEY in .env).']];
const GUIDE=[
 'Put the green ball (or any bright object) in the gripper, then <b>click it in the overhead image</b>. A green circle confirms the target is locked.',
 'Press <b>Start calibration</b> &mdash; the leader arm now drives the follower. Move it gently by hand.',
 'Move the ball somewhere over the mat, hold it still, press <b>Capture</b> (spacebar).',
 'Repeat at <b>6+ spots spread across the whole camera view</b> &mdash; not a line, aim for the corners (numbered dots + coverage % show on the image; the fitted cm grid appears live after 4 samples &mdash; check it lines up with the table). Residual under 5 mm is good: <span id="g-res">no fit yet</span>.',
 'Rest the fingertip on the table and press <b>Touch table</b> (records the table height).',
 'Press <b>Finish</b> to save. <b>Cancel</b> writes nothing.'];
let ACT={},actsig='',S={},curTab='setup';
let ring=null,frameW=640,frameH=480,haveWH=false,dropZone=null,zoneSig='',rulesSig='',logSig='',tickerSig='';
// Frame size for px scaling: prefer the live size from /state (haveWH) — an <img> on an mjpeg stream can keep
// a stale naturalWidth from before a server restart at another resolution, mis-scaling clicks and the ring.
function frameWH(im){return haveWH?[frameW,frameH]:[im.naturalWidth||frameW,im.naturalHeight||frameH];}
let lastSay='',lastCallSig='',lastErrSig='',targetLocked=false,ckDismissed=false;
function toast(msg,kind){const t=document.createElement('div');t.className='toast'+(kind?' '+kind:'');
 t.textContent=msg;$('toasts').appendChild(t);
 while($('toasts').children.length>4)$('toasts').firstChild.remove();
 setTimeout(()=>t.remove(),kind==='say'?6000:4500);}
function helpOf(n){return (ACT[n]&&ACT[n].help)||'';}
function showMsg(btn,j){const s=btn&&btn.closest('.sec');const m=s&&s.querySelector('.rowmsg');
 if(m){m.textContent=j.message||'';m.classList.toggle('err',!j.ok);}}
async function act(name,body,btn){
 if(btn){btn.classList.add('busy');delete btn.dataset.stat;}
 let j;
 try{
  const r=await fetch('/action/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  j=await r.json();
 }catch(e){j={ok:false,message:''+e};}
 if(btn){btn.classList.remove('busy');btn.dataset.stat=j.ok?'ok':'err';setTimeout(()=>{delete btn.dataset.stat;},1400);}
 if(!j.ok)toast((j.message||name+' failed'),'err');
 if(btn)showMsg(btn,j);
 loadActions();
 return j;}
// ---------- build the panel from GET /actions ----------
function numify(v){return /^-?\d+(\.\d+)?$/.test(v)?parseFloat(v):v;}
function genericRow(a,showName){const ins=(a.params||[]).map(p=>{const d=p.default==null?'':String(p.default);
 const w=(p.name==='text'||p.name==='task')?' class="grow"':'';
 return '<input'+w+' data-p="'+esc(p.name)+'" placeholder="'+esc(p.name)+'" value="'+esc(d)+'">';}).join('');
 return '<div class="arow">'+(showName?'<span class="an">'+esc(a.name)+' &middot; '+esc(a.group)+'</span>':'')+
  '<button class="btn" data-n="'+esc(a.name)+'" title="'+esc(a.help||'')+'">'+esc(a.label||a.name)+'</button>'+ins+
  '<div class="help">'+esc(a.help||'')+'</div></div>';}
function wireRows(root){root.querySelectorAll('button[data-n]').forEach(b=>b.onclick=()=>{const body={};
 b.parentElement.querySelectorAll('input').forEach(i=>{const v=i.value.trim();if(v!=='')body[i.dataset.p]=numify(v);});
 act(b.dataset.n,body,b);});}
function buildConn(){$('connrows').innerHTML=DEVICES.map(([k,a,t,d])=>
 '<div class="crow"><span class="cn">'+esc(t)+'</span><span class="cst" data-ck="'+k+'">off</span>'+
 '<button class="btn" data-ca="'+esc(a)+'" data-ck="'+k+'" title="'+esc(helpOf(a))+'">Connect</button>'+
 '<span class="cd">'+esc(d)+'</span><div class="help">'+esc(helpOf(a))+'</div></div>').join('');
 $('connrows').querySelectorAll('button[data-ca]').forEach(b=>b.onclick=()=>{
  const on=!!(((S.run&&S.run.connected)||{})[b.dataset.ck]);
  act(b.dataset.ca,{connect:!on},b);});}
function buildCalib(){
 $('calguide').innerHTML=GUIDE.map(g=>'<li>'+g+'</li>').join('');
 const order=['calib_start','calib_capture','calib_touch','calib_undo','calib_finish','calib_cancel'];
 const rows=order.filter(n=>ACT[n]&&ACT[n].label).map(n=>ACT[n]);
 $('calbtns').innerHTML=rows.map(a=>
  '<button class="btn'+(a.name==='calib_start'?' primary':'')+(a.name==='calib_capture'?' big':'')+'" data-n="'+esc(a.name)+
  '" title="'+esc(a.help||'')+'">'+esc(a.label)+(a.name==='calib_capture'?' (space)':'')+'</button>').join('')+
  rows.map(a=>'<div class="help">'+esc(a.label)+': '+esc(a.help||'')+'</div>').join('');
 wireRows($('calbtns'));
 const fb=$('calbtns').querySelector('button[data-n="calib_finish"]');
 if(fb){let armed=false;fb.onclick=async()=>{
  const j=await act('calib_finish',armed?{force:true}:{},fb);
  if(!j.ok&&j.data&&j.data.force_needed){armed=true;fb.textContent='Finish anyway (low coverage)';}
  else{armed=false;fb.textContent='Finish';}};}}
function buildJog(){const JOGS=[['x','+'],['x','-'],['y','+'],['y','-'],['z','+'],['z','-'],['roll','+'],['roll','-']];
 $('jogpad').innerHTML=JOGS.map(([ax,sg])=>'<button class="btn" data-ax="'+ax+'" data-sg="'+sg+'">'+ax+' '+sg+'</button>').join('');
 $('jogpad').querySelectorAll('button').forEach(b=>b.onclick=()=>{
  const d=Math.abs(parseFloat($('jogd').value)||10)*(b.dataset.sg==='-'?-1:1);
  act('jog',{axis:b.dataset.ax,delta:d},b);});}
function buildExtras(){TABS.forEach(t=>{$('extra-'+t).innerHTML='';});
 for(const n in ACT){const a=ACT[n];
  if(!a.label||CLAIMED.has(a.name))continue;
  const t=TABOF[a.group]||'debug';
  $('extra-'+t).insertAdjacentHTML('beforeend',genericRow(a,true));}
 TABS.forEach(t=>wireRows($('extra-'+t)));}
function buildRaw(){const all=Object.values(ACT);
 $('rawlist').innerHTML=all.length?all.map(a=>genericRow(a,true)).join(''):'<span class="empty">no actions registered</span>';
 wireRows($('rawlist'));}
function applyHelp(){document.querySelectorAll('[data-h]').forEach(el=>{el.textContent=helpOf(el.dataset.h);});
 document.querySelectorAll('[data-h-title]').forEach(el=>{const h=helpOf(el.dataset.hTitle);if(h)el.title=h;});}
function renderAll(){
 const has=n=>!!ACT[n];
 $('sec-mode').hidden=!has('connect_robot');
 $('sec-calib').hidden=!has('calib_start');
 $('sec-task').hidden=!has('set_task');
 $('sec-run').hidden=!has('start');
 $('sec-voice').hidden=!has('say_to_bot');
 $('sec-rules').hidden=!has('add_rule');
 $('sec-models').hidden=!has('get_models');
 $('sec-zones').hidden=!has('set_zone_drop');
 $('sec-robot').hidden=!has('home');
 $('sec-log').hidden=!has('log_clear');
 $('mictoggle').hidden=!has('mic_on');
 if(has('connect_robot'))buildConn();
 if(has('calib_start'))buildCalib();
 if(has('home'))buildJog();
 buildExtras();buildRaw();applyHelp();
 if(curTab==='tune'&&has('get_models')&&!window._models)loadModels();}
async function loadActions(){try{const a=await(await fetch('/actions')).json();
 const sig=JSON.stringify(a);if(sig===actsig)return;actsig=sig;
 ACT={};a.forEach(x=>{ACT[x.name]=x;});renderAll();}catch(e){}}
// ---------- tabs / help toggle ----------
function setTab(t){curTab=t;
 document.querySelectorAll('.tabbar .tb').forEach(b=>b.classList.toggle('sel',b.dataset.tab===t));
 TABS.forEach(x=>{$('tab-'+x).hidden=x!==t;});
 if(t==='tune'&&ACT.get_models&&!window._models)loadModels();
 if(t==='debug'){logSig='';loadLog();}}
document.querySelectorAll('.tabbar .tb').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
$('helpbtn').onclick=()=>{document.body.classList.toggle('showhelp');$('helpbtn').classList.toggle('sel');};
$('conns').onclick=()=>setTab('setup');
$('estop').onclick=()=>act('torque_off',{},$('estop'));
$('torqon').onclick=()=>act('torque_on',{},$('torqon'));
$('demobtn').onclick=()=>openDemo();
$('d-close').onclick=()=>closeDemo();
$('d-estop').onclick=()=>act('torque_off',{},$('d-estop'));
$('ckx').onclick=()=>{ckDismissed=true;$('checklist').hidden=true;};
$('ck1').onclick=e=>{e.preventDefault();setTab('setup');};
$('ck2').onclick=e=>{e.preventDefault();setTab('setup');};
$('ck3').onclick=e=>{e.preventDefault();setTab('operate');};
$('banner').onclick=()=>{const t=$('banner').dataset.tab;if(t)setTab(t);};
// ---------- fixed controls ----------
$('taskbtn').onclick=()=>act('set_task',{text:$('task').value.trim()},$('taskbtn'));
$('b-start').onclick=()=>act('start',{},$('b-start'));
$('b-pause').onclick=()=>act('pause',{},$('b-pause'));
$('b-resume').onclick=()=>act('resume',{},$('b-resume'));
$('b-stop').onclick=()=>act('stop',{},$('b-stop'));
$('b-step').onclick=()=>act('step_once',{},$('b-step'));
$('b-maxsteps').onclick=()=>{const n=parseInt($('maxsteps').value,10);
 if(!n){showMsg($('b-maxsteps'),{ok:false,message:'enter a number of steps first'});return;}
 act('set_max_steps',{n:n},$('b-maxsteps'));};
$('corrbtn').onclick=async()=>{const t=$('corr').value.trim();
 if(!t){showMsg($('corrbtn'),{ok:false,message:'type a correction first'});return;}
 const j=await act('say_to_bot',{text:t},$('corrbtn'));if(j.ok)$('corr').value='';};
$('corr').onkeydown=e=>{if(e.key==='Enter')$('corrbtn').click();};
$('newrule').onkeydown=e=>{if(e.key==='Enter')$('rulebtn').click();};
$('rulebtn').onclick=async()=>{const t=$('newrule').value.trim();
 if(!t){showMsg($('rulebtn'),{ok:false,message:'type a rule first'});return;}
 rulesSig='';const j=await act('add_rule',{text:t},$('rulebtn'));if(j.ok)$('newrule').value='';};
$('b-clearhints').onclick=()=>act('clear_hints',{},$('b-clearhints'));
$('b-home').onclick=()=>act('home',{},$('b-home'));
$('b-open').onclick=()=>act('open_gripper',{},$('b-open'));
$('b-close').onclick=()=>act('close_gripper',{},$('b-close'));
$('b-ton').onclick=()=>act('torque_on',{},$('b-ton'));
$('b-goto').onclick=()=>{const x=parseFloat($('gx').value),y=parseFloat($('gy').value),z=parseFloat($('gz').value);
 if([x,y,z].some(v=>isNaN(v))){showMsg($('b-goto'),{ok:false,message:'fill in x, y and z (table mm) first'});return;}
 act('goto',{x:x,y:y,z:z},$('b-goto'));};
$('b-logclear').onclick=async()=>{await act('log_clear',{},$('b-logclear'));logSig='';loadLog();};
$('mictoggle').onclick=async()=>{const on=S.voice&&S.voice.listening;
 const j=on?await act('mic_off',{},$('mictoggle')):await act('mic_on',{},$('mictoggle'));
 $('micnote').textContent=j.message||'';tick();};
// ---------- overhead click: zone drop placement or calibration target pick ----------
$('ov').onclick=async e=>{const im=$('ov'),r=im.getBoundingClientRect();
 const [nw,nh]=frameWH(im);
 const u=Math.round((e.clientX-r.left)/r.width*nw),v=Math.round((e.clientY-r.top)/r.height*nh);
 if(dropZone){const z=dropZone;dropZone=null;zoneSig='';
  const j=await act('px_to_mm',{u:u,v:v});
  if(j.ok&&j.data)await act('set_zone_drop',{name:z,x:j.data.x,y:j.data.y});
  return;}
 const cal=S.calibration;
 if(ACT.calib_sample&&(curTab==='setup'||(cal&&cal.state==='running'))){
  const j=await act('calib_sample',{u:u,v:v});
  targetLocked=!!(j.ok&&j.data&&j.data.det);
  ring=(j.data&&j.data.det)?j.data.det:null;drawRing();
  const m=$('msg-calib');if(m){m.textContent=j.message||'';m.classList.toggle('err',!j.ok);}}};
function drawRing(){const im=$('ov'),el=$('ring');if(!ring){el.style.display='none';return;}
 const [nw,nh]=frameWH(im),sx=im.clientWidth/nw,sy=im.clientHeight/nh,u=ring[0],v=ring[1];
 const rad=Math.min(ring[2],Math.min(nw,nh)/2);  // cap: never draw beyond the image box
 el.style.display='block';
 el.style.left=Math.max(0,(u-rad)*sx)+'px';el.style.top=Math.max(0,(v-rad)*sy)+'px';
 el.style.width=Math.min(2*rad*sx,im.clientWidth)+'px';el.style.height=Math.min(2*rad*sy,im.clientHeight)+'px';}
window.addEventListener('resize',drawRing);
// ---------- push to talk ----------
let rec=null,chunks=[];
function b64(buf){const u=new Uint8Array(buf);let t='';for(let i=0;i<u.length;i+=0x8000)t+=String.fromCharCode.apply(null,u.subarray(i,i+0x8000));return btoa(t);}
async function pttDown(){if(rec)return;
 if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia||typeof MediaRecorder==='undefined'){
  $('micnote').textContent='microphone unavailable in this browser context (use the text box instead)';return;}
 try{const st=await navigator.mediaDevices.getUserMedia({audio:true});
  rec=new MediaRecorder(st,MediaRecorder.isTypeSupported&&MediaRecorder.isTypeSupported('audio/webm')?{mimeType:'audio/webm'}:undefined);
  chunks=[];rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  rec.onstop=async()=>{st.getTracks().forEach(t=>t.stop());const b=new Blob(chunks,{type:rec.mimeType||'audio/webm'});rec=null;pttIdle();
   if(b.size<200){$('micnote').textContent='clip too short, try again';return;}
   $('transcript').textContent='transcribing...';
   const j=await act('transcribe',{audio_b64:b64(await b.arrayBuffer()),mime:b.type||'audio/webm'});
   $('transcript').textContent=j.ok&&j.data&&j.data.text?('heard: "'+j.data.text+'"'):('!! '+(j.message||'transcription failed'));};
  rec.start();$('ptt').classList.add('rec');$('ptt').textContent='recording... release to send';$('micnote').textContent='';}
 catch(e){$('micnote').textContent='mic unavailable: '+e+' (use the text box instead)';rec=null;pttIdle();}}
function pttIdle(){$('ptt').classList.remove('rec');$('ptt').innerHTML='&#127908; Hold to talk';}
function pttUp(){if(rec&&rec.state!=='inactive')rec.stop();else pttIdle();}
$('ptt').onmousedown=pttDown;$('ptt').onmouseup=pttUp;$('ptt').onmouseleave=pttUp;
$('ptt').ontouchstart=e=>{e.preventDefault();pttDown();};
$('ptt').ontouchend=e=>{e.preventDefault();pttUp();};
// ---------- models ----------
const PROVIDERS=[['openai','vision model'],['elevenlabs_tts','tts model'],['elevenlabs_stt','stt model'],['elevenlabs_voice','voice']];
async function loadModels(){const j=await act('get_models');if(j.ok&&j.data){window._models=j.data;renderModels();}
 else $('mrows').innerHTML='<span class="empty">'+esc(j.message||'get_models failed')+'</span>';}
function renderModels(){const d=window._models;if(!d)return;const cur=d.current||{};
 const opts={openai:d.openai||[],elevenlabs_tts:(d.elevenlabs||{}).tts||[],elevenlabs_stt:(d.elevenlabs||{}).stt||[],
  elevenlabs_voice:((d.elevenlabs||{}).voices||[]).map(v=>[v.id,v.name+' ('+v.id.slice(0,8)+')'])};
 $('mrows').innerHTML=PROVIDERS.map(([p,lbl])=>{let os=opts[p].map(o=>{const pair=Array.isArray(o)?o:[o,o];
   return '<option value="'+esc(pair[0])+'"'+(pair[0]===cur[p]?' selected':'')+'>'+esc(pair[1])+'</option>';}).join('');
  if(cur[p]&&!opts[p].some(o=>(Array.isArray(o)?o[0]:o)===cur[p]))os='<option value="'+esc(cur[p])+'" selected>'+esc(cur[p])+'</option>'+os;
  return '<div class="mrow"><span class="lbl2">'+esc(lbl)+'</span><select data-prov="'+esc(p)+'" title="'+esc(helpOf('set_model'))+'">'+os+'</select></div>';}).join('');
 $('mnotes').textContent=(d.notes||[]).join('; ');
 $('mrows').querySelectorAll('select').forEach(sl=>sl.onchange=async()=>{
  const j=await act('set_model',{provider:sl.dataset.prov,value:sl.value});
  showMsg(sl,j);window._models=null;loadModels();});}
// ---------- state-driven rendering ----------
const fmt=v=>v==null?'-':(typeof v==='number'?v.toFixed(0):v);
function dis(btn,reason){if(!btn)return;btn.disabled=!!reason;
 btn.title=reason?reason:(btn.dataset.hTitle?helpOf(btn.dataset.hTitle):btn.title);}
function bannerInfo(){
 const r=S.run,rb=S.robot,cal=S.calibration;
 if(rb&&rb.torque===false)return['E-STOP - torque is OFF. Press "Torque on" to re-enable the arm.','err','debug'];
 if(cal&&cal.state==='running'){const n=cal.n||0;
  if(!cal.det&&n===0&&!targetLocked)return['Calibrating - click the target ball in the overhead image to lock its colour.','amber','setup'];
  if(n<4)return['Calibrating - move the ball, hold still, press Capture (space): '+n+' of 4+ samples.','amber','setup'];
  return['Calibrating - '+n+' samples, residual '+(cal.residual_mean_mm==null?'n/a':cal.residual_mean_mm+' mm')+'. Spread more spots, then Touch table and Finish.','amber','setup'];}
 if(cal&&cal.state==='error')return['Calibration error - '+(cal.message||''),'err','setup'];
 if(!r)return[cal?'Calibration HUD - follow the steps in Setup.':'Waiting for the server...','','setup'];
 const ph=r.phase,c=r.connected||{};
 if(ph==='running'){let h='';
  if(rb&&rb.holding)h=' - holding: '+rb.holding;
  return['Sorting - step '+r.step+'/'+r.max_steps+h+(S.status?(' - '+S.status):''),'ok','operate'];}
 if(ph==='paused')return['Paused at step '+r.step+' - Resume (key: p) to continue.','amber','operate'];
 if(ph==='error')return['Error - '+(r.last_error||'see the Debug log'),'err','debug'];
 if(ph==='done')return['Done - '+(r.result||'finished')+' Start again from Operate.','ok','operate'];
 if(ph==='stopped')return['Stopped'+(r.result?' - '+r.result:'')+'. Start again from Operate.','','operate'];
 const miss=['robot','cams','vlm'].filter(k=>!c[k]);
 if(miss.length)return['Not connected ('+miss.join(', ')+') - connect the devices in Setup.','amber','setup'];
 if(S.perception&&S.perception.calibrated===false)return['Not calibrated - open Setup and run the camera calibration.','amber','setup'];
 return['Ready - set a task and press Start in Operate.','ok','operate'];}
function renderChecklist(){
 const r=S.run;const el=$('checklist');
 if(!r||ckDismissed){el.hidden=true;return;}
 if(r.phase!=='idle'){el.hidden=true;return;}
 el.hidden=false;
 const c=r.connected||{},cal=S.calibration;
 const d1=!!(c.robot&&c.cams&&c.vlm);
 const d2=!!(S.perception&&S.perception.calibrated)||!!(cal&&cal.state==='fitted');
 const cur=!d1?1:(!d2?2:3);
 [[1,d1],[2,d1&&d2],[3,false]].forEach(([i,done])=>{const st=$('ck'+i);
  st.classList.toggle('done',!!done);st.classList.toggle('cur',i===cur);});}
function renderGuide(cal){
 const lis=$('calguide').children;if(!lis.length)return;
 let cur=-1;const done=[];
 if(cal&&cal.state==='running'){const n=cal.n||0,locked=targetLocked||!!cal.det||n>0;
  done.push(0);if(locked)done.push(1);
  if(!locked)cur=0;
  else if(n===0)cur=2;
  else if(n<6){done.push(2);cur=3;}
  else if(cal.z_offset_mm==null){done.push(2,3);cur=4;}
  else{done.push(2,3,4);cur=5;}}
 else if(cal&&cal.state==='fitted'){for(let i=0;i<6;i++)done.push(i);}
 else cur=targetLocked?1:0;
 for(let i=0;i<lis.length;i++){lis[i].classList.toggle('cur',i===cur);
  lis[i].classList.toggle('done',done.includes(i)&&i!==cur);}
 const g=$('g-res');
 if(g)g.textContent=cal&&cal.residual_mean_mm!=null?('residual mean '+cal.residual_mean_mm+' / max '+cal.residual_max_mm+' mm'):'no fit yet';}
function renderRules(rl){if(!rl)return;const sig=JSON.stringify(rl);if(sig===rulesSig)return;rulesSig=sig;
 const list=Array.isArray(rl)?rl:(rl.list||[]);
 $('rlist').innerHTML=list.map((r,i)=>'<li><span class="rt">'+esc(r)+'</span>'+
  '<a data-mv="'+i+'|up" title="move up (higher priority)">&#8593;</a><a data-mv="'+i+'|down" title="move down">&#8595;</a>'+
  '<a data-del="'+i+'" title="delete this rule">&#10005;</a></li>').join('')
  ||'<li style="background:none;border:none"><span class="empty">No rules yet - type one below, e.g. "red things go in LEFT".</span></li>';
 $('rlist').querySelectorAll('a[data-mv]').forEach(a=>a.onclick=()=>{const p=a.dataset.mv.split('|');rulesSig='';act('move_rule',{i:+p[0],dir:p[1]});});
 $('rlist').querySelectorAll('a[data-del]').forEach(a=>a.onclick=()=>{rulesSig='';act('delete_rule',{i:+a.dataset.del});});
 const hints=(rl&&rl.hints)||[];
 $('hints').textContent=hints.length?hints.join(' | '):'none';}
function renderPerc(pc){if(!pc)return;
 const sig=JSON.stringify([pc.zones,dropZone]);if(sig===zoneSig)return;zoneSig=sig;
 $('zrows').innerHTML=(pc.zones||[]).map(z=>'<div class="zrow"><span class="zn">'+esc(z.name)+'</span><span>drop ('+z.drop[0]+', '+z.drop[1]+')</span>'+
  '<button class="btn sm'+(dropZone===z.name?' arm':'')+'" data-z="'+esc(z.name)+'">'+(dropZone===z.name?'click the image...':'set drop')+'</button></div>').join('')
  ||'<span class="empty">no zones configured</span>';
 $('zrows').querySelectorAll('button[data-z]').forEach(b=>b.onclick=()=>{dropZone=dropZone===b.dataset.z?null:b.dataset.z;zoneSig='';renderPerc(S.perception);});}
function updateEnables(){
 const r=S.run;if(!r)return;
 const c=r.connected||{},ph=r.phase,running=ph==='running',paused=ph==='paused',active=running||paused;
 const miss=['robot','cams','vlm'].filter(k=>!c[k]);
 const needs='needs '+miss.join(' + ')+' connected - connect the devices in Setup';
 const toff=S.robot&&S.robot.torque===false;
 dis($('b-start'),active?'already running - Stop first':(miss.length?needs:null));
 dis($('b-pause'),running?null:'no run in progress');
 dis($('b-resume'),!paused?'nothing is paused':(toff?'torque is off (E-STOP) - press Torque on first':null));
 dis($('b-stop'),active?null:'no run to stop');
 dis($('b-step'),(running&&!paused)?'pause first to single-step':(miss.length&&!active?needs:null));
 $('why-run').textContent=(!active&&miss.length)?('Start '+needs):(paused&&toff?'Resume is blocked: torque is off (E-STOP) - press Torque on first':'');
 const noRobot=!c.robot;
 const rWhy=noRobot?'needs a robot connected - connect it in Setup':
  (running&&!paused?'the sorting loop is running - pause it first':(toff?'torque is off (E-STOP) - press Torque on':null));
 ['b-home','b-open','b-close','b-goto'].forEach(id=>dis($(id),rWhy));
 const jw=noRobot?rWhy:(running&&!paused?rWhy:(toff?rWhy:null));
 document.querySelectorAll('#jogpad button').forEach(b=>dis(b,jw));
 dis($('b-ton'),noRobot?'needs a robot connected - connect it in Setup':null);
 $('why-robot').textContent=rWhy||'';
 const cal=S.calibration,calRun=cal&&cal.state==='running';
 $('calbtns').querySelectorAll('button[data-n]').forEach(b=>{const n=b.dataset.n;
  if(n==='calib_start')dis(b,calRun?'calibration is already running':(noRobot&&ACT.connect_robot?'needs a robot connected - step 1 above':null));
  else dis(b,calRun?null:'press Start calibration first');});
 $('why-calib').textContent=calRun?'':(noRobot&&ACT.connect_robot?'Calibration needs a robot - finish step 1 above.':'');
 $('connrows').querySelectorAll('button[data-ca]').forEach(b=>{const on=!!c[b.dataset.ck];
  b.textContent=on?'Disconnect':'Connect';b.classList.toggle('primary',!on);
  dis(b,active?'stop the run before connecting or disconnecting':(calRun?'finish or cancel the calibration first':null));});
 $('connrows').querySelectorAll('.cst').forEach(el=>{const on=!!c[el.dataset.ck];
  el.textContent=on?'connected':'off';el.classList.toggle('on',on);});}
async function tick(){let s;
 try{s=await(await fetch('/state')).json();}catch(e){return;}
 S=s;
 const r=s.run,rb=s.robot,cal=s.calibration;
 if(s.frame_wh){frameW=s.frame_wh[0];frameH=s.frame_wh[1];haveWH=true;}
 // image-age chip: how old the overhead frame is on the server (fresh < 2s, stale in red)
 const fa=s.frame_age_s;
 $('imgage').textContent=fa==null?'no image yet':('image '+fa.toFixed(1)+'s old');
 $('imgage').className=fa==null?'stale':(fa<2?'':(fa<6?'warn':'stale'));
 // header: connection dots
 const conn=(r&&r.connected)||{};
 [['robot','cd-robot'],['cams','cd-cams'],['vlm','cd-vlm']].forEach(([k,id])=>{const el=$(id);
  el.classList.toggle('on',!!conn[k]);el.querySelector('.cv').textContent=conn[k]?'on':'off';});
 // banner
 const b=bannerInfo();
 $('banner').textContent=b[0];$('banner').title=b[0];$('banner').className=b[1];$('banner').dataset.tab=b[2];
 // E-STOP / torque
 const toff=rb&&rb.torque===false;
 $('estop').classList.toggle('off',!!toff);
 $('estop').textContent=toff?'TORQUE OFF':'E-STOP';
 $('torqon').hidden=!toff;
 // mic chip + toggle
 const listening=!!(s.voice&&s.voice.listening);
 $('micchip').hidden=!listening;
 $('mictoggle').textContent=listening?'● Listening: ON':'Listening: OFF';
 $('mictoggle').classList.toggle('live',listening);
 renderChecklist();
 // readout bar (fixed-height, mono: no layout shift)
 const p=(rb&&rb.ee_pose)||s.ee_pose||{};
 const jd=rb&&rb.joints_deg?rb.joints_deg.map(x=>x.toFixed(0)).join(' '):'-';
 const grip=rb?(rb.gripper_open?'open':'closed'):'-';
 const vs=s.vlm;
 const vtxt=vs?(vs.model+' '+(vs.last_latency_ms==null?'':vs.last_latency_ms+'ms')+
  (vs.last_cost_usd!=null?' ~$'+vs.last_cost_usd.toFixed(4):'')):'-';
 $('readout').innerHTML=
  '<b>EE</b> x '+fmt(p.x)+'  y '+fmt(p.y)+'  z '+fmt(p.z)+'  roll '+fmt(p.roll_deg)+'   <b>joints</b> '+esc(jd)+
  '\n<b>gripper</b> '+grip+'   <b>holding</b> '+esc((rb?rb.holding:s.holding)??'nothing')+
  '   <b>step latency</b> '+(s.latency_ms==null?'-':s.latency_ms+'ms')+'   <b>vlm</b> '+esc(vtxt)+
  '   <b>updated</b> '+(s.age_s==null?'-':s.age_s+'s ago');
 if(vs)$('vlmstat').textContent=vs.model+'  '+(vs.last_latency_ms==null?'- ms':vs.last_latency_ms+' ms')+
  (vs.last_cost_usd!=null?'  ~$'+vs.last_cost_usd.toFixed(4)+'/call':'')+
  (vs.last_usage?'  ('+vs.last_usage.input_tokens+' in / '+vs.last_usage.output_tokens+' out)':'');
 // calibration card + guide + ring
 if(cal&&cal.state){
  const res=cal.residual_mean_mm!=null?('residual mean '+cal.residual_mean_mm+' / max '+cal.residual_max_mm+' mm'):'no fit yet (4+ samples)';
  const tgt=cal.target||{};
  $('calstat').textContent='state '+cal.state+'   samples '+(cal.n??0)+'   '+res+
   '\ntarget '+(tgt.name||'-')+'   z offset '+(cal.z_offset_mm==null?'not measured':cal.z_offset_mm.toFixed(1)+' mm')+
   (cal.coverage_pct!=null&&cal.state==='running'?('\ncoverage '+cal.coverage_pct+'% - '+(cal.coverage_verdict||'')):'')+
   '\n'+(cal.message||'')+(cal.loaded?('\n'+cal.loaded):'');
  renderGuide(cal);
  if(cal.state==='running'){ring=cal.det||null;drawRing();}}
 else if(!$('sec-calib').hidden){$('calstat').textContent='no calibration session';renderGuide(null);}
 // task + steps
 if(r){$('taskcur').textContent=r.task||'(default: sort sensibly)';
  $('maxsteps').placeholder=r.max_steps??'40';}
 // voice
 if(s.voice){$('vqueue').textContent=(s.voice.queue||[]).join(' | ')||'none';
  if(s.voice.last_transcript&&!$('transcript').textContent.startsWith('!!')&&$('transcript').textContent!=='transcribing...')
   $('transcript').textContent='heard: "'+s.voice.last_transcript+'"';}
 renderRules(s.rules);
 renderPerc(s.perception);
 updateEnables();
 // toasts: bot speech, failures, run errors
 if(s.say&&s.say!==lastSay){lastSay=s.say;toast('“'+s.say+'”','say');}
 const lc=s.last_call;
 const lcs=lc?JSON.stringify(lc):'';
 if(lcs&&lcs!==lastCallSig){lastCallSig=lcs;
  if(lc.result&&/^FAILED/.test(lc.result))toast(lc.tool+': '+lc.result,'err');}
 if(r&&r.last_error&&r.last_error!==lastErrSig){lastErrSig=r.last_error;toast(r.last_error,'err');}
 // demo view
 if(!$('demo').hidden){
  $('d-banner').textContent=b[0];
  $('d-say').textContent=s.say?'“'+s.say+'”':'';
  $('d-decision').textContent=lc?(lc.tool+'('+JSON.stringify(lc.args||{})+')  →  '+(lc.result||'')):'no decisions yet';
  const rl=s.rules,list=(rl&&rl.list)||[];
  const tsig=list.join('|');
  if(tsig!==tickerSig){tickerSig=tsig;
   $('d-ticker').textContent=list.length?('RULES:  '+list.join('   •   ')):'no rules yet - add some in Operate';}}
}
// ---------- decision log ----------
async function loadLog(){if(curTab!=='debug'||$('sec-log').hidden)return;
 try{const l=await(await fetch('/log')).json();
  const sig=l.length?l[0].i+'/'+l.length:'0';if(sig===logSig)return;logSig=sig;
  $('logbox').innerHTML=l.map(e=>'<div class="logrow'+(e.ok===false?' bad':'')+'">'+
   (e.thumb_b64?'<img src="data:image/jpeg;base64,'+e.thumb_b64+'">':'')+
   '<div class="lr"><span class="lt">#'+e.step+' '+new Date(e.t*1000).toLocaleTimeString()+(e.latency_ms!=null?' '+e.latency_ms+'ms':'')+'</span>\n'+
   esc(e.tool)+'('+esc(JSON.stringify(e.args||{}))+')\n'+esc(e.result||'')+(e.say?'\n"'+esc(e.say)+'"':'')+'</div></div>').join('')
   ||'<span class="empty">no decisions yet - start a run in Operate</span>';}catch(e){}}
// ---------- demo view ----------
function openDemo(){$('d-ov').src='/overhead.mjpg';tickerSig='';$('demo').hidden=false;}
function closeDemo(){$('demo').hidden=true;$('d-ov').src='';}
// ---------- keyboard ----------
document.addEventListener('keydown',e=>{
 const tag=e.target&&e.target.tagName;
 if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT')return;
 const k=e.key;
 if(k==='e'||k==='E')act('torque_off');
 else if(k==='p'||k==='P'){const ph=S.run&&S.run.phase;
  if(ph==='paused')act('resume');else if(ph==='running')act('pause');}
 else if(k===' '){const cal=S.calibration;
  if(cal&&cal.state==='running'){e.preventDefault();act('calib_capture');}}
 else if(k==='Escape')closeDemo();});
loadActions();setInterval(loadActions,5000);
tick();setInterval(tick,500);
setInterval(loadLog,1500);
</script></body></html>
"""


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
        self._t_frame = 0.0  # when the overhead frame was last refreshed (drives the image-age chip)
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
                self._t_frame = time.time()
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
            s["frame_age_s"] = round(time.time() - self._t_frame, 1) if self._t_frame else None
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
                                "latency_ms": 1234, "last_call": {"tool": "pick_at", "args": {"x_cm": 25.0, "y_cm": 12.0}},
                                "say": "picking the red block", "rules": ["red things go left"],
                                "voice_queue": ["put round things in the middle"]})
        base = f"http://127.0.0.1:{port}"
        post = lambda path, body: json.load(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
            base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"), timeout=3))
        with urllib.request.urlopen(base + "/", timeout=3) as r:
            assert r.status == 200 and b"overhead.mjpg" in r.read()
        with urllib.request.urlopen(base + "/state", timeout=3) as r:
            s = json.load(r)
            assert s["step"] == 1 and s["ee_pose"]["x"] == 150 and s["frame_wh"] == [640, 360]
            assert s["frame_age_s"] is not None and s["frame_age_s"] < 60, s["frame_age_s"]
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
