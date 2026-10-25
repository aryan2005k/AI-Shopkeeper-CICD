let products = [];
const charts = {};
const esc = v => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const $ = id => document.getElementById(id);
const pct = v => v == null ? '—' : (v * 100).toFixed(2) + '%';
const num = (v, d = 3) => v == null ? '—' : Number(v).toFixed(d);

async function api(url, options = {}) {
  const r = await fetch(url, options);
  const data = await r.json().catch(() => ({detail: 'Server returned an invalid response.'}));
  if (!r.ok) throw new Error(data.detail || data.message || `Request failed (${r.status})`);
  return data;
}

function busy(btn, on, label) {
  if (on) { btn.dataset.label = btn.textContent; btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>${esc(label)}`; }
  else { btn.disabled = false; btn.textContent = btn.dataset.label || btn.textContent; }
}

/* ---------- charts: Chart.js if it loaded, otherwise a small canvas fallback ---------- */
function drawFallback(canvas, labels, series) {
  const dpr = window.devicePixelRatio || 1, w = canvas.parentElement.clientWidth - 4 || 600, h = Number(canvas.dataset.h) || 260;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const pad = {l: 44, r: 12, t: 14, b: 26};
  const all = series.flatMap(s => s.data).filter(v => v != null && isFinite(v));
  const max = Math.max(...all, 1) * 1.1;
  const x = i => pad.l + (labels.length < 2 ? 0 : i * (w - pad.l - pad.r) / (labels.length - 1));
  const y = v => pad.t + (h - pad.t - pad.b) * (1 - v / max);
  ctx.font = '11px Arial'; ctx.fillStyle = '#687386'; ctx.strokeStyle = '#e6eaf1'; ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = max * k / 4, yy = y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillText(Math.round(v), 6, yy + 4);
  }
  ctx.fillText(labels[0] || '', pad.l, h - 8);
  const last = labels[labels.length - 1] || ''; ctx.fillText(last, w - pad.r - ctx.measureText(last).width, h - 8);
  series.forEach((s, si) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
    let pen = false;
    s.data.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      if (!pen) { ctx.moveTo(x(i), y(v)); pen = true; } else ctx.lineTo(x(i), y(v));
    });
    ctx.stroke(); ctx.setLineDash([]);
    if (series.length > 1) { ctx.fillStyle = s.color; ctx.fillText(s.label, pad.l + 8 + si * 110, 12); }
  });
}

function renderChart(id, labels, series, opts = {}) {
  const canvas = $(id); if (!canvas) return;
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
  if (typeof Chart === 'undefined') { drawFallback(canvas, labels, series); return; }
  charts[id] = new Chart(canvas, {
    type: 'line',
    data: {labels, datasets: series.map(s => ({label: s.label, data: s.data, borderColor: s.color, backgroundColor: s.color,
      borderDash: s.dash || [], pointRadius: s.points ? 3 : 0, borderWidth: 2, tension: .25, fill: false, spanGaps: false}))},
    options: {responsive: true, animation: false, interaction: {mode: 'index', intersect: false},
      scales: {x: {display: opts.showX !== false, ticks: {maxTicksLimit: 8}}, y: {beginAtZero: true}},
      plugins: {legend: {display: series.length > 1}}}
  });
}

/* ---------- dashboard / products ---------- */
async function loadAll() {
  try {
    const a = await api('/api/analytics');
    $('units').textContent = Math.round(a.total_units).toLocaleString();
    $('revenue').textContent = '₹' + a.revenue.toLocaleString(undefined, {maximumFractionDigits: 2});
    $('productCount').textContent = a.product_count;
    $('topProducts').innerHTML = a.top_products.map((x, i) =>
      `<div class="metric"><b>${i + 1}. ${esc(x.product_name)}</b>${Number(x.qty).toFixed(0)} units</div>`).join('');
    renderChart('trendChart', a.trend.map(x => x.date), [{label: 'Units Sold', data: a.trend.map(x => x.qty), color: '#2f5bea'}], {showX: false});
    products = await api('/api/products');
    $('productTable').innerHTML = products.map(p => `
      <tr>
        <td><b>${esc(p.product_name)}</b><br><small>${esc(p.product_id)}</small></td>
        <td>${esc(p.category)}</td><td>₹${Number(p.unit_price).toFixed(2)}</td>
        <td><input class="inv" data-id="${esc(p.product_id)}" data-field="current_stock" type="number" min="0" step="0.01" value="${Number(p.current_stock)}"></td>
        <td><input class="inv" data-id="${esc(p.product_id)}" data-field="incoming_stock" type="number" min="0" step="0.01" value="${Number(p.incoming_stock)}"></td>
        <td><input class="inv" data-id="${esc(p.product_id)}" data-field="safety_stock" type="number" min="0" step="0.01" value="${Number(p.safety_stock)}"></td>
        <td><button class="small" onclick="saveInventory('${esc(p.product_id)}')">Save</button></td>
      </tr>`).join('');
    const sel = $('productSelect'), keep = sel.value;
    sel.innerHTML = products.map(p => `<option value="${esc(p.product_id)}">${esc(p.product_name)} (${esc(p.product_id)})</option>`).join('');
    if (keep) sel.value = keep;
  } catch (e) { console.error(e); $('forecastResult').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
}

async function saveInventory(product_id) {
  const vals = {product_id};
  document.querySelectorAll(`.inv[data-id="${CSS.escape(product_id)}"]`).forEach(x => vals[x.dataset.field] = Number(x.value));
  try { await api('/api/inventory', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(vals)}); setStatus(`Inventory saved for ${product_id}.`); }
  catch (e) { setStatus(e.message, true); }
}
function setStatus(msg, error = false) { const el = $('inventoryStatus'); el.textContent = msg; el.className = 'status ' + (error ? 'error' : 'success'); }

async function uploadCSV() {
  const f = $('csvFile').files[0]; if (!f) { alert('Choose a CSV file first.'); return; }
  const fd = new FormData(); fd.append('file', f);
  const btn = $('uploadBtn'), st = $('uploadStatus'); busy(btn, true, 'Importing…'); st.className = 'status'; st.textContent = '';
  try {
    const d = await api('/api/upload-csv', {method: 'POST', body: fd});
    st.className = 'status success';
    st.textContent = `${d.message} ${d.rows_imported} rows (${d.date_from} to ${d.date_to}), ${d.products_affected} products` +
      (d.rows_skipped ? `, ${d.rows_skipped} invalid rows skipped.` : '.');
    await loadAll();
  } catch (e) { st.className = 'status error'; st.textContent = e.message; }
  finally { busy(btn, false); }
}

/* ---------- forecasting ---------- */
function forecastChart(d) {
  const hist = d.history, fc = d.forecast;
  const labels = hist.map(x => x.date).concat(fc.map(x => x.date));
  const nulls = n => Array(n).fill(null), h = hist.length, lastV = hist[h - 1].quantity;
  const join = arr => nulls(h - 1).concat([lastV], arr);
  renderChart('forecastChart', labels, [
    {label: 'Actual', data: hist.map(x => x.quantity).concat(nulls(fc.length)), color: '#172033'},
    {label: 'Forecast', data: join(fc.map(x => x.predicted_demand)), color: '#2f5bea'},
    {label: 'Lower 80%', data: join(fc.map(x => x.lower)), color: '#9db4f5', dash: [5, 4]},
    {label: 'Upper 80%', data: join(fc.map(x => x.upper)), color: '#9db4f5', dash: [5, 4]}
  ]);
}

async function runForecast() {
  const product_id = $('productSelect').value, horizon = Number($('horizon').value);
  if (!product_id) { alert('Import sales data first.'); return; }
  const body = {product_id, horizon, retrain: $('optRetrain').checked, tune: $('optTune').checked,
                lead_time_days: Number($('leadTime').value), service_level: Number($('serviceLevel').value)};
  const box = $('forecastResult'), btn = $('forecastBtn');
  busy(btn, true, body.tune ? 'Tuning & training…' : 'Forecasting…');
  box.innerHTML = '<p class="muted2">First run trains the models; later runs reuse them until the data change.</p>';
  try {
    const d = await api('/api/forecast', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const inv = d.inventory, rows = d.model_results;
    box.innerHTML = `
      ${d.warnings.map(w => `<div class="warnbox">⚠ ${esc(w)}</div>`).join('')}
      <div class="resultGrid">
        <div class="metric">Best Model<b>${esc(d.best_model)}</b></div>
        <div class="metric">Total Forecast (${d.forecast.length} days)<b>${d.total_predicted_demand.toFixed(2)}</b></div>
        <div class="metric">Reorder Quantity<b>${d.recommendation.reorder_quantity.toFixed(2)}</b></div>
      </div>
      <div class="warning">${esc(d.recommendation.status)}</div>
      <p class="note">Selected by ${esc(d.selection_method)}. ${d.cached ? 'Loaded from cache' : 'Trained now'} (${esc(d.trained_at)})${d.tuned ? ', tuned' : ''}.
        ${d.total_days} days of history, ${d.test_rows} hold-out days.</p>
      <div class="chartbox"><canvas id="forecastChart" data-h="280"></canvas><p class="note">${esc(d.interval_note)}</p></div>
      <h4>Forecast</h4>
      <table class="forecastTable"><thead><tr><th>Date</th><th>Predicted Demand</th><th>80% range</th></tr></thead>
      <tbody>${d.forecast.map(x => `<tr><td>${x.date}</td><td>${x.predicted_demand}</td><td>${x.lower} – ${x.upper}</td></tr>`).join('')}</tbody></table>
      <h4>Model Evaluation</h4>
      <div class="tablewrap"><table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>WAPE (hold-out)</th><th>WAPE (CV)</th><th>WAPE (7-day recursive)</th></tr></thead>
      <tbody>${rows.map(x => `<tr class="${x.model === d.best_model ? 'best' : ''}"><td>${esc(x.model)}${x.model === d.best_model ? '<span class="badge">selected</span>' : ''}</td>
        <td>${num(x.mae)}</td><td>${num(x.rmse)}</td><td>${pct(x.wape)}</td><td>${pct(x.cv_wape)}</td><td>${pct(x.wape_7d)}</td></tr>`).join('')}</tbody></table></div>
      <p class="note">${esc(d.evaluation_note)}</p>
      <h4>Why this forecast? (${esc(d.explanation_method)})</h4>
      ${d.explanation.length ? d.explanation.map(x => `<span class="pill">${esc(x.feature)}: ${x.importance.toFixed(4)}</span> `).join('') : '<p class="muted2">No feature explanation for baseline models.</p>'}
      <h4>Inventory</h4>
      <p>Current ${inv.current_stock}, incoming ${inv.incoming_stock}, safety ${inv.safety_stock}, projected after forecast <b>${inv.projected_stock_after_forecast}</b>.</p>
      <p class="note">Statistical safety-stock suggestion for a ${inv.lead_time_days}-day lead time at ${(inv.service_level * 100).toFixed(0)}% service level:
        <b>${inv.suggested_safety_stock}</b> units (you can enter it in the table above).</p>`;
    forecastChart(d);
  } catch (e) { box.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  finally { busy(btn, false); }
}

async function runLSTM() {
  const product_id = $('productSelect').value, horizon = Number($('horizon').value);
  if (!product_id) { alert('Import sales data first.'); return; }
  const box = $('lstmResult'), btn = $('lstmBtn');
  busy(btn, true, 'Running LSTM…');
  box.innerHTML = '<p class="muted2">Training the first time can take a minute; later runs reuse the saved model.</p>';
  try {
    const d = await api('/api/forecast-lstm', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({product_id, horizon, retrain: $('optRetrain').checked})});
    box.innerHTML = `<h4>LSTM Results ${d.metrics.cached ? '<span class="badge">cached</span>' : ''}</h4>
      <div class="resultGrid"><div class="metric">MAE<b>${num(d.metrics.mae)}</b></div><div class="metric">RMSE<b>${num(d.metrics.rmse)}</b></div><div class="metric">WAPE<b>${pct(d.metrics.wape)}</b></div></div>
      <p class="note">Trained ${d.metrics.epochs_run ?? '?'} epochs (early stopping), ${d.metrics.trained_rows} sequences, ${d.metrics.test_rows} hold-out days.</p>
      <table><thead><tr><th>Date</th><th>Predicted Demand</th></tr></thead><tbody>${d.forecast.map(x => `<tr><td>${x.date}</td><td>${x.predicted_demand}</td></tr>`).join('')}</tbody></table>
      <h4>LSTM vs the other models (hold-out WAPE)</h4>
      <table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>WAPE</th></tr></thead><tbody>
      ${d.comparison.map(x => `<tr class="${x.model === 'LSTM' ? 'best' : ''}"><td>${esc(x.model)}</td><td>${num(x.mae)}</td><td>${num(x.rmse)}</td><td>${pct(x.wape)}</td></tr>`).join('')}</tbody></table>
      <p class="note">${esc(d.note)}</p>`;
  } catch (e) {
    box.innerHTML = `<p class="error">${esc(e.message)}</p><p class="muted2">The LSTM needs the optional TensorFlow package: <code>pip install -r requirements-lstm.txt</code>.</p>`;
  } finally { busy(btn, false); }
}

window.addEventListener('load', loadAll);   // after Chart.js (local or CDN) has had a chance to load
