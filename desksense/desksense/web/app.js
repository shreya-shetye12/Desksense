'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const palette = ['#6b9582', '#91a66c', '#bd9b6b', '#c27f6e', '#6d91a4', '#a18cac', '#b29469', '#a19d85'];
  const state = { file: null, url: null, result: null, health: null, busy: false, elapsed: 0, selected: -1, timer: null, dragDepth: 0, exportUrl: null, tooLong: false };
  const video = $('video');
  const chart = $('confidenceChart');
  const maxBytes = 100 * 1024 * 1024;
  let toastTimer;

  const fmtTime = value => {
    const n = Number.isFinite(value) ? Math.max(0, value) : 0;
    return `${String(Math.floor(n / 60)).padStart(2, '0')}:${(n % 60).toFixed(1).padStart(4, '0')}`;
  };
  const pct = value => `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
  const colorFor = label => palette[Math.max(0, state.result?.classes.indexOf(label) ?? 0) % palette.length];
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  };
  const svg = (tag, attrs = {}, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text != null) node.textContent = text;
    return node;
  };

  function toast(text) {
    clearTimeout(toastTimer);
    $('toast').textContent = text;
    $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
  }

  function message(text, retry = false) {
    const box = $('message');
    box.replaceChildren();
    box.hidden = !text;
    if (!text) return;
    box.append(el('span', '', text));
    if (retry) {
      const button = el('button', '', 'Reconnect');
      button.type = 'button';
      button.addEventListener('click', connect);
      box.append(button);
    }
  }


  function validateSettings() {
    const windowSize = Number($('windowInput').value);
    const stride = Number($('strideInput').value);
    const alpha = Number($('alphaInput').value);
    if (!Number.isFinite(windowSize) || windowSize <= 0 || windowSize > 30) throw new Error('Choose a window length greater than 0 and no more than 30 seconds.');
    if (!Number.isFinite(stride) || stride <= 0 || stride > windowSize) throw new Error('Window stride must be greater than 0 and no longer than the window length.');
    if (!Number.isFinite(alpha) || alpha <= 0 || alpha > 1) throw new Error('Smoothing must be greater than 0 and no more than 1.');
    return { window: windowSize, stride, alpha };
  }

  function updateButton() {
    let invalid = false;
    try { validateSettings(); } catch { invalid = true; }
    $('analyzeButton').disabled = state.busy || !state.file || !state.health || invalid || state.tooLong;
    if (state.busy) return;
    $('analyzeText').textContent = state.result ? 'Analyze again' : 'Analyze video';
    if (state.tooLong) { $('setupNote').textContent = 'Trim this video to 2 minutes or less before analysis.'; return; }
    $('setupNote').textContent = !state.health ? 'Connect to the local server to run inference.' : !state.file ? 'Choose a video to start your analysis.' : invalid ? 'Stride must be positive and no longer than the window.' : 'Processed locally. Your upload is removed after analysis.';
  }

  async function connect() {
    try {
      const response = await fetch('/health', { cache: 'no-store', signal: AbortSignal.timeout(10000) });
      if (!response.ok) throw new Error('The model is not ready.');
      const health = await response.json();
      if (!Array.isArray(health.classes) || !health.classes.length) throw new Error('The server did not return a valid class list.');
      state.health = health;
      $('connectionStatus').className = 'connection ready';
      $('connectionStatus').replaceChildren(el('i'), document.createTextNode('Model connected'));
      $('modelName').textContent = health.model_name || (health.model === 'videomae' ? 'VideoMAE' : health.model === 'frame_mean' ? 'Frame-mean ResNet-18' : 'Video classifier');
      $('modelDetails').textContent = `${health.classes.length} interaction classes · sliding windows`;
      $('deviceBadge').textContent = String(health.device || 'local').toUpperCase();
      $('sampleButton').hidden = !health.sample_available;
      message('');
    } catch {
      state.health = null;
      $('connectionStatus').className = 'connection offline';
      $('connectionStatus').replaceChildren(el('i'), document.createTextNode('Disconnected'));
      $('modelDetails').textContent = 'Local model is unavailable';
      message('The local inference server is unavailable. Check the Terminal running FastAPI, then reconnect.', true);
    }
    updateButton();
  }

  function clearResult() {
    state.result = null;
    if (state.exportUrl) { URL.revokeObjectURL(state.exportUrl); state.exportUrl = null; }
    $('exportButton').removeAttribute('href');
    state.selected = -1;
    $('resultsPanel').hidden = true;
    $('sessionSummary').hidden = true;
    $('currentPrediction').hidden = true;
    $('predictionEmpty').hidden = false;
    $('predictionEmptyText').textContent = 'The current action will appear here as you explore your results.';
    $('videoOverlay').hidden = true;
    $('chartEmpty').hidden = false;
    $('actionTrack').replaceChildren();
    $('trackLegend').replaceChildren();
    $('resultsBody').replaceChildren();
    $('chartFootnote').textContent = 'Analyze your clip, then click the timeline to explore. Scores are not verified probabilities of correctness.';
    renderChart();
  }

  function acceptFile(file) {
    if (state.busy) return toast('Wait for this analysis to finish before replacing the video.');
    if (!file) return;
    if (file.size === 0) return message('This file is empty. Choose a video with content.');
    if (file.size > maxBytes) return message('This video exceeds the 100 MiB upload limit. Choose a smaller clip.');
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|webm|m4v|avi|mkv)$/i.test(file.name)) return message('Choose a video file, such as MP4, MOV, or WebM.');
    message('');
    video.pause();
    if (state.url) URL.revokeObjectURL(state.url);
    state.file = file;
    state.tooLong = false;
    state.url = URL.createObjectURL(file);
    clearResult();
    $('dropZone').hidden = true;
    $('videoStage').hidden = false;
    $('fileInfo').hidden = false;
    $('replaceButton').hidden = false;
    $('sourceTag').hidden = true;
    $('fileName').textContent = file.name;
    $('fileName').title = file.name;
    $('fileSize').textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MiB`;
    $('videoDuration').textContent = 'Reading video…';
    video.src = state.url;
    video.load();

    updateButton();
  }

  function validateResult(data) {
    if (!data || !Array.isArray(data.classes) || data.classes.length < 2 || !data.classes.every(c => typeof c === 'string') || !Number.isFinite(data.duration) || data.duration <= 0 || !Array.isArray(data.predictions) || !data.predictions.length) throw new Error('The server returned an incomplete prediction result.');
    let lastEnd = -1;
    for (const p of data.predictions) {
      if (!Number.isFinite(p.start) || !Number.isFinite(p.end) || p.start < 0 || p.end <= p.start || p.end < lastEnd || p.end > data.duration + .05 || !Number.isFinite(p.available_at) || p.available_at < p.end - .05 || !data.classes.includes(p.label) || !data.classes.includes(p.raw_label)) throw new Error('The server returned invalid prediction windows.');
      if (!Number.isFinite(p.confidence) || p.confidence < 0 || p.confidence > 1 || !Number.isFinite(p.raw_confidence) || p.raw_confidence < 0 || p.raw_confidence > 1 || !Array.isArray(p.probabilities) || p.probabilities.length !== data.classes.length || !p.probabilities.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw new Error('The server returned invalid confidence scores.');
      lastEnd = p.end;
    }
    return data;
  }

  function setBusy(busy) {
    state.busy = busy;
    $('settingsFields').disabled = busy;
    $('replaceButton').disabled = busy;
    $('sampleButton').disabled = busy;
    $('analyzeButton').classList.toggle('loading', busy);
    $('analyzeText').textContent = busy ? 'Analyzing video' : state.result ? 'Analyze again' : 'Analyze video';
    $('processingStatus').hidden = !busy;
    if (busy) $('setupNote').textContent = 'Inference speed depends on clip length and your hardware.';
    updateButton();
  }

  async function analyze() {
    if (!state.file || state.busy || !state.health) return;
    let settings;
    try { settings = validateSettings(); } catch (error) { return message(error.message); }
    message('');
    video.pause();
    clearResult();
    setBusy(true);

    const start = performance.now();
    $('processingText').textContent = 'Uploading & analyzing…';
    $('processingTimer').textContent = '0s';
    state.timer = setInterval(() => { $('processingTimer').textContent = `${((performance.now() - start) / 1000).toFixed(0)}s`; }, 500);
    try {
      const response = await fetch(`/predict?${new URLSearchParams(settings)}`, { method: 'POST', body: state.file, headers: { 'Content-Type': 'application/octet-stream' } });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : response.status === 429 ? 'The inference worker is busy. Wait for its current request to finish and try again.' : 'Analysis failed. Check the server output and try another clip.');
      state.result = validateResult(data);
      state.elapsed = (performance.now() - start) / 1000;
      $('resultsPanel').hidden = false;
      $('sessionSummary').hidden = false;
      $('chartEmpty').hidden = true;
      $('summaryWindows').textContent = String(data.predictions.length);
      $('summaryElapsed').textContent = `${state.elapsed.toFixed(1)}s`;
      $('summaryDuration').textContent = `${data.duration.toFixed(2)}s`;
      $('resultCount').textContent = String(data.predictions.length);
      $('chartFootnote').textContent = `Each prediction becomes available at its window end. First output: ${data.predictions[0].available_at.toFixed(2)}s. Streaming scores are heuristic; whole-clip calibration does not validate window or smoothed confidence.`;
      prepareExport();
      renderResults();
      renderChart();
      renderTrack();
      updatePlayhead();

      toast(`Analysis complete · ${data.predictions.length} prediction windows`);
    } catch (error) {
      message(error instanceof TypeError ? 'Could not reach the local server. Check that FastAPI is running, then try again.' : error.message);

    } finally {
      clearInterval(state.timer);
      state.timer = null;
      setBusy(false);
    }
  }

  function chartGeometry() {
    return { left: 42, right: 882, top: 13, bottom: 142, duration: state.result?.duration || (Number.isFinite(video.duration) ? video.duration : 10) };
  }

  function renderChart() {
    const { left, right, top, bottom, duration } = chartGeometry();
    chart.replaceChildren(svg('title', { id: 'chartTitle' }, 'Model confidence timeline'), svg('desc', { id: 'chartDescription' }, state.result ? 'Top-class smoothed model scores at each window end. Click the timeline or use the table to seek the video.' : 'Analyze a video to see model scores over time.'));
    const defs = svg('defs');
    const gradient = svg('linearGradient', { id: 'confidenceFill', x1: 0, y1: 0, x2: 0, y2: 1 });
    gradient.append(svg('stop', { offset: '0%', 'stop-color': '#dce9cc', 'stop-opacity': '.6' }), svg('stop', { offset: '100%', 'stop-color': '#f4f8ed', 'stop-opacity': '.05' }));
    defs.append(gradient); chart.append(defs);
    for (const value of [0, .25, .5, .75, 1]) {
      const y = bottom - value * (bottom - top);
      chart.append(svg('line', { x1: left, y1: y, x2: right, y2: y, class: 'chart-grid' }), svg('text', { x: left - 10, y: y + 3, 'text-anchor': 'end', class: 'chart-axis-label' }, `${value * 100}%`));
    }
    for (let i = 0; i <= 4; i++) {
      const t = i * duration / 4;
      chart.append(svg('text', { x: left + (right - left) * i / 4, y: 164, 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle', class: 'chart-axis-label' }, `${t.toFixed(duration < 10 ? 1 : 0)}s`));
    }
    const predictions = state.result?.predictions || [];
    if (predictions.length) {
      const point = (p, raw = false) => [left + Math.min(p.available_at, duration) / duration * (right - left), bottom - (raw ? p.raw_confidence : p.confidence) * (bottom - top)];
      const points = predictions.map(p => point(p));
      const path = pts => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
      chart.append(svg('path', { d: `${path(points)} L${points.at(-1)[0]},${bottom} L${points[0][0]},${bottom} Z`, class: 'chart-area' }));
      if ($('showRaw').checked) chart.append(svg('path', { d: path(predictions.map(p => point(p, true))), class: 'chart-raw' }));
      chart.append(svg('path', { d: path(points), class: 'chart-line' }));
      points.forEach(([x, y], i) => chart.append(svg('circle', { cx: x, cy: y, r: 3.4, class: 'chart-point', 'data-index': i })));
      chart.append(svg('line', { id: 'chartPlayhead', x1: left, x2: left, y1: top, y2: bottom, class: 'chart-playhead' }));
      const hit = svg('rect', { x: left, y: top, width: right - left, height: bottom - top, class: 'chart-hit' });
      hit.addEventListener('click', event => {
        const bounds = chart.getBoundingClientRect();
        const x = (event.clientX - bounds.left) / bounds.width * 900;
        seek(Math.max(0, Math.min(duration, (x - left) / (right - left) * duration)));
      });
      chart.append(hit);
    }
    $('rawLegend').hidden = !$('showRaw').checked;
    updateChartPlayhead();
  }

  function seek(time) {
    if (!state.url) return;
    try { video.currentTime = Math.max(0, Math.min(Number.isFinite(video.duration) ? video.duration : time, time)); } catch { toast('Your browser cannot seek this video format.'); }
    updatePlayhead();
  }

  function renderTrack() {
    const track = $('actionTrack');
    const legend = $('trackLegend');
    track.replaceChildren(); legend.replaceChildren();
    const result = state.result;
    if (!result) return;
    const used = new Set();
    result.predictions.forEach((p, index) => {
      const next = result.predictions[index + 1]?.available_at ?? result.duration;
      const span = Math.max(0, next - p.available_at);
      if (span > 0) {
        const button = el('button');
        button.type = 'button';
        button.style.left = `${p.available_at / result.duration * 100}%`;
        button.style.width = `${span / result.duration * 100}%`;
        button.style.backgroundColor = colorFor(p.label);
        button.title = `${fmtTime(p.available_at)} · ${p.label}`;
        button.setAttribute('aria-label', `Seek to ${p.available_at.toFixed(2)} seconds: ${p.label}`);
        button.addEventListener('click', () => seek(p.available_at));
        track.append(button);
      }
      used.add(p.label);
    });
    used.forEach(label => {
      const item = el('span');
      const dot = el('i'); dot.style.backgroundColor = colorFor(label);
      item.append(dot, document.createTextNode(label)); legend.append(item);
    });
  }

  function renderResults() {
    const body = $('resultsBody'); body.replaceChildren();
    state.result.predictions.forEach((p, index) => {
      const row = el('tr'); row.dataset.index = index;
      const timeCell = el('td');
      const button = el('button', 'time-button', `${p.start.toFixed(2)}–${p.end.toFixed(2)}s`);
      button.type = 'button'; button.setAttribute('aria-label', `Inspect window ${index + 1}, ${p.label}, available at ${p.available_at.toFixed(2)} seconds`);
      button.addEventListener('click', () => seek(p.available_at)); timeCell.append(button);
      const actionCell = el('td'); const action = el('span', 'prediction-cell'); const dot = el('i'); dot.style.backgroundColor = colorFor(p.label); action.append(dot, document.createTextNode(p.label)); actionCell.append(action);
      row.append(timeCell, actionCell, el('td', '', pct(p.confidence)), el('td', 'raw-column', `${p.raw_label} · ${pct(p.raw_confidence)}`));
      body.append(row);
    });
  }

  function updateChartPlayhead() {
    const line = $('chartPlayhead');
    if (!line || !state.result) return;
    const { left, right, duration } = chartGeometry();
    const x = left + Math.min(video.currentTime || 0, duration) / duration * (right - left);
    line.setAttribute('x1', x); line.setAttribute('x2', x);
    chart.querySelectorAll('.chart-point').forEach(point => {
      const active = Number(point.dataset.index) === state.selected;
      point.classList.toggle('active', active); point.setAttribute('r', active ? '4.8' : '3.4');
    });
  }

  function updatePlayhead() {
    const time = video.currentTime || 0;
    $('playheadTime').textContent = fmtTime(time);
    const result = state.result;
    if (!result) return;
    // Causal selection: a later window cannot be shown before its end/availability.
    let selected = -1;
    result.predictions.forEach((p, i) => { if (p.available_at <= time + .001) selected = i; });
    state.selected = selected;
    $('resultsBody').querySelectorAll('tr').forEach(row => row.classList.toggle('active', Number(row.dataset.index) === selected));
    updateChartPlayhead();
    $('currentPrediction').hidden = selected < 0;
    $('predictionEmpty').hidden = selected >= 0;
    $('videoOverlay').hidden = selected < 0;
    if (selected < 0) {
      $('predictionEmptyText').textContent = `The first prediction is available at ${result.predictions[0].available_at.toFixed(2)}s. Press play or select a window below.`;
      return;
    }
    const p = result.predictions[selected];
    $('predictionLabel').textContent = p.label;
    $('predictionColor').style.backgroundColor = colorFor(p.label);
    $('predictionScore').textContent = pct(p.confidence);
    $('scoreMeter').style.width = `${p.confidence * 100}%`;
    $('rawSummary').textContent = `Raw: ${p.raw_label} · ${pct(p.raw_confidence)}`;
    $('overlayLabel').textContent = p.label;
    $('overlayScore').textContent = pct(p.confidence);
    $('rejectionNote').textContent = result.threshold > 0 ? p.abstain ? 'Below the saved rejection threshold. Review this prediction.' : 'Above the saved threshold; this does not establish correctness.' : 'Rejection is off for this run. Low-scoring predictions are still returned.';
    const ranks = result.classes.map((label, i) => ({ label, score: p.probabilities[i] })).filter(r => r.label !== p.label).sort((a, b) => b.score - a.score).slice(0, 3);
    $('classRankings').replaceChildren(...ranks.map(rank => {
      const item = el('div', 'ranking-row'); item.append(el('span', '', rank.label), el('strong', '', pct(rank.score))); return item;
    }));
  }

  function prepareExport() {
    if (!state.result) return;
    if (typeof state.result.download_url === 'string' && /^\/exports\/[A-Za-z0-9_-]+\.json$/.test(state.result.download_url)) {
      $('exportButton').href = state.result.download_url;
    } else {
      const blob = new Blob([JSON.stringify(state.result, null, 2) + '\n'], { type: 'application/json' });
      state.exportUrl = URL.createObjectURL(blob);
      $('exportButton').href = state.exportUrl;
    }
    $('exportButton').download = `${state.file.name.replace(/\.[^.]+$/, '')}_desksense.json`;
  }

  $('chooseButton').addEventListener('click', () => $('fileInput').click());
  $('replaceButton').addEventListener('click', () => { if (!state.busy) $('fileInput').click(); });
  $('fileInput').addEventListener('change', event => { acceptFile(event.target.files[0]); event.target.value = ''; });
  $('analyzeButton').addEventListener('click', analyze);
  $('exportButton').addEventListener('click', () => toast('JSON download requested'));
  $('showRaw').addEventListener('change', renderChart);
  ['windowInput', 'strideInput', 'alphaInput'].forEach(id => $(id).addEventListener('input', () => {
    $('alphaValue').textContent = Number($('alphaInput').value).toFixed(2);
    updateButton();
    if (state.result && !$('analyzeButton').disabled) $('setupNote').textContent = 'Settings changed. Analyze again to update your predictions.';
  }));
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => document.addEventListener(name, event => event.preventDefault()));
  const dropSurface = $('sourceHeading').closest('section');
  dropSurface.addEventListener('dragenter', () => { state.dragDepth++; $('dropZone').classList.add('dragging'); });
  dropSurface.addEventListener('dragleave', () => { if (--state.dragDepth <= 0) { state.dragDepth = 0; $('dropZone').classList.remove('dragging'); } });
  dropSurface.addEventListener('drop', event => { state.dragDepth = 0; $('dropZone').classList.remove('dragging'); acceptFile(event.dataTransfer.files[0]); });
  video.addEventListener('loadedmetadata', () => {
    $('videoDuration').textContent = Number.isFinite(video.duration) ? `${video.duration.toFixed(2)}s` : 'Duration unavailable';
    state.tooLong = video.duration > 120;
    updateButton();
    if (state.tooLong && !state.busy) message('This video is longer than 2 minutes. Trim it before analysis; the server enforces a 120-second limit.');
    renderChart(); updatePlayhead();
  });
  video.addEventListener('error', () => { if (state.file) message('Your browser cannot preview this video format. The server may still be able to analyze it; MP4/H.264 is a useful preview alternative.'); });
  ['timeupdate', 'seeked', 'ended'].forEach(name => video.addEventListener(name, updatePlayhead));
  $('sampleButton').addEventListener('click', async () => {
    if (state.busy) return;
    $('sampleButton').disabled = true;
    try {
      const response = await fetch('/sample-video');
      if (!response.ok) throw new Error('The optional sample clip is unavailable. Choose your own video instead.');
      const blob = await response.blob();
      acceptFile(new File([blob], 'sample-interaction.webm', { type: blob.type || 'video/webm' }));
    } catch (error) { message(error.message); }
    finally { $('sampleButton').disabled = false; }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !$('analyzeButton').disabled) analyze();
  });
  window.addEventListener('beforeunload', () => { if (state.url) URL.revokeObjectURL(state.url); if (state.exportUrl) URL.revokeObjectURL(state.exportUrl); });
  renderChart();
  connect();
})();
