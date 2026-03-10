async function jget(url){
  const r = await fetch(url, {cache:"no-store"});
  if(!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}
async function jpost(url, body){
  const r = await fetch(url, {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body || {})
  });
  const j = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
  return j;
}

function fmt(x, d=4){
  if(x===null || x===undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if(!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}
function fmtPrice(x){
  if(x===null || x===undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if(!Number.isFinite(n)) return "—";
  if(n >= 100) return n.toFixed(2);
  if(n >= 1) return n.toFixed(4);
  return n.toFixed(6);
}
function clsForRisk(r){
  if(r==="OK") return "good";
  if(r==="CAUTION") return "warn";
  if(r==="HIGH") return "bad";
  return "muted";
}

let STATUS=null;
let SCORES=[];
let TRAINING=null;

function setText(id, text){
  const el = document.getElementById(id);
  if(el) el.textContent = text;
}
function setHTML(id, html){
  const el = document.getElementById(id);
  if(el) el.innerHTML = html;
}

function renderStatus(){
  if(!STATUS) return;
  const demo = STATUS.demo_mode ? "DEMO" : "LIVE";
  const model = (STATUS.model && STATUS.model.status) ? STATUS.model.status.toUpperCase() : "—";
  setText("badge-mode", demo);
  setText("badge-model", model);

  const cbok = STATUS.coinbase.ok;
  setHTML("coinbase-ok", cbok ? `<span class="good">OK</span>` : `<span class="warn">Not confirmed</span>`);
  const cbmsg = STATUS.coinbase.last_error ? STATUS.coinbase.last_error : (STATUS.coinbase.last_ok_utc ? `Last OK: ${STATUS.coinbase.last_ok_utc}` : "No recent calls");
  setText("coinbase-msg", cbmsg);

  const cov = STATUS.coverage || {};
  setText("coverage-main", `${cov.scored_count||0} scored / ${cov.universe_count||0} in universe`);
  const skips = cov.top_skip_reasons ? Object.entries(cov.top_skip_reasons).slice(0,4).map(([k,v])=>`${k}:${v}`).join("  ") : "—";
  setText("coverage-skip", `Skips: ${skips}`);

  const cfg = STATUS.config || {};
  setText("scan-times", `Last scan: ${cov.last_run_utc || "—"}`);
  setText("scan-config", `Interval ${cfg.scan_interval_minutes}m · Horizon ${cfg.horizon_minutes}m · Target ${(cfg.target_pct*100).toFixed(1)}% · Benchmark ${cfg.benchmark_symbol}`);

  const rl = STATUS.coinbase.rate_limit_stats || {};
  setText("rl-main", `${rl.ok||0}/${rl.requests||0} ok · 429:${rl.http_429||0} · 5xx:${rl.http_5xx||0}`);
  setText("rl-sub", `Retries:${rl.retries||0} · Backoff:${(rl.backoff_seconds_total||0).toFixed(1)}s`);

  // training
  if(TRAINING){
    renderTraining();
  }
}

function uniqueQuotes(rows){
  const s = new Set();
  rows.forEach(r=>{ if(r.quote) s.add(r.quote); });
  return Array.from(s).sort();
}

function ensureQuoteOptions(){
  const sel = document.getElementById("quote-filter");
  if(!sel) return;
  const existing = new Set(Array.from(sel.options).map(o=>o.value));
  const quotes = uniqueQuotes(SCORES);
  quotes.forEach(q=>{
    if(!existing.has(q)){
      const opt = document.createElement("option");
      opt.value = q;
      opt.textContent = q;
      sel.appendChild(opt);
    }
  });
}

function filteredRows(){
  const minProb = Number(document.getElementById("min-prob").value || "0");
  const quote = document.getElementById("quote-filter").value || "";
  const showStale = document.getElementById("show-stale").checked;

  return SCORES.filter(r=>{
    const p = Number(r.prob_3);
    if(Number.isFinite(minProb) && p < minProb) return false;
    if(quote && r.quote !== quote) return false;
    const isStale = (r.notes || "").includes("STALE_CANDLES");
    if(!showStale && isStale) return false;
    return true;
  });
}

function renderScores(){
  const body = document.getElementById("scores-body");
  if(!body) return;
  const rows = filteredRows();

  if(rows.length===0){
    body.innerHTML = `<tr><td colspan="7" class="muted">No rows match your filters.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map(r=>{
    const src = r.prob_3_source==="model" ? "model" : "heuristic";
    const notes = (r.notes || "");
    return `
      <tr>
        <td><b>${r.product}</b><div class="muted" style="font-size:12px">${src}</div></td>
        <td class="num">${fmtPrice(r.price)}</td>
        <td class="num"><b>${fmt(r.prob_3, 4)}</b></td>
        <td>${r.quote || "—"}</td>
        <td>${r.last_candle_time || "—"}</td>
        <td>${notes || "—"}</td>
        <td class="${clsForRisk(r.risk)}"><b>${r.risk || "—"}</b></td>
      </tr>
    `;
  }).join("");
}

function renderTraining(){
  const el = document.getElementById("training-status");
  if(!el || !TRAINING) return;
  if(TRAINING.running){
    const p = TRAINING.progress || {};
    const msg = TRAINING.last_error ? `Error: ${TRAINING.last_error}` : "";
    el.textContent = `Running · stage=${TRAINING.stage || "—"} · ${p.done||0}/${p.total||0} · rows=${TRAINING.rows||"—"} ${msg}`;
  }else{
    if(TRAINING.stage==="error"){
      el.textContent = `Error: ${TRAINING.last_error || "Unknown"}`;
    }else if(TRAINING.stage==="done"){
      const m = TRAINING.metrics ? JSON.stringify(TRAINING.metrics) : "";
      el.textContent = `Done · rows=${TRAINING.rows||"—"} · finished=${TRAINING.finished_at_utc||"—"}`;
    }else{
      el.textContent = "Idle.";
    }
  }
}

async function refreshAll(){
  try{
    STATUS = await jget("/api/status");
    renderStatus();
  }catch(e){
    console.error(e);
  }
  try{
    const j = await jget("/api/scores");
    SCORES = j.rows || [];
    ensureQuoteOptions();
    renderScores();
  }catch(e){
    console.error(e);
  }
}

async function pollTraining(){
  try{
    TRAINING = await jget("/api/training/status");
    renderTraining();
    if(TRAINING.running){
      setTimeout(pollTraining, 2000);
    }
  }catch(e){
    console.error(e);
  }
}

document.addEventListener("DOMContentLoaded", ()=>{
  document.getElementById("btn-refresh").addEventListener("click", refreshAll);
  document.getElementById("min-prob").addEventListener("input", renderScores);
  document.getElementById("quote-filter").addEventListener("change", renderScores);
  document.getElementById("show-stale").addEventListener("change", renderScores);

  document.getElementById("btn-train").addEventListener("click", async ()=>{
    const pw = document.getElementById("admin-password").value;
    try{
      await jpost("/train", {password: pw});
      TRAINING = {running:true, stage:"starting", progress:{done:0,total:0}};
      renderTraining();
      pollTraining();
    }catch(e){
      alert(e.message || String(e));
    }
  });

  refreshAll().then(()=>pollTraining());
});
