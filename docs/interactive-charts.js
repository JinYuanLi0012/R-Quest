'use strict';

(() => {
  const ns = 'http://www.w3.org/2000/svg';
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const make = (tag, attrs, parent, text) => {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    if (parent) parent.append(el);
    return el;
  };
  const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');

  // Round scores come from the original PDF paths, not pixel estimates.
  const chart = document.getElementById('trajectory-chart');
  const tx = round => 64 + round * 43.2;
  const ty = score => 354 - (score - 32) / 23 * 302;
  [35, 40, 45, 50, 55].forEach(score => {
    const y = ty(score);
    make('line', { x1: 64, x2: 496, y1: y, y2: y, class: 'chart-grid' }, chart);
    make('text', { x: 52, y: y + 6, 'text-anchor': 'end', class: 'chart-tick' }, chart, score);
  });
  [0, 2, 4, 6, 8, 10].forEach(round => make('text', { x: tx(round), y: 382, 'text-anchor': 'middle', class: 'chart-tick' }, chart, round));
  make('text', { x: 280, y: 416, 'text-anchor': 'middle', class: 'chart-axis-title' }, chart, 'Self-evolution round');
  make('text', { transform: 'translate(20 205) rotate(-90)', 'text-anchor': 'middle', class: 'chart-axis-title' }, chart, 'Math average (%)');
  const band = make('path', { class: 'trajectory-band' }, chart);
  const zeroPath = make('path', { class: 'trajectory-line zero-line' }, chart);
  const questPath = make('path', { class: 'trajectory-line quest-line' }, chart);
  const points = make('g', {}, chart);
  const roundDots = TRAJECTORY_DATA.map(d => ({
    q: make('circle', { cx: tx(d.round), cy: ty(d.rquest), r: 4.4, class: 'quest-dot' }, points),
    z: make('circle', { cx: tx(d.round), cy: ty(d.rzero), r: 3.6, class: 'zero-dot' }, points)
  }));
  const guide = make('line', { y1: 48, y2: 354, class: 'chart-guide' }, chart);
  const qHead = make('circle', { r: 6, class: 'quest-dot chart-head' }, chart);
  const zHead = make('circle', { r: 5, class: 'zero-dot chart-head' }, chart);
  const qEnd = make('text', { x: 504, y: ty(51.92) + 5, class: 'trajectory-end quest-label' }, chart, '51.92');
  const zEnd = make('text', { x: 504, y: ty(34.60) + 5, class: 'trajectory-end zero-label' }, chart, '34.60');
  const robotQ = make('image', { href: 'assets/trajectory-quest.png', x: 427, y: ty(51.92) - 54, width: 49, height: 49, class: 'trajectory-robot' }, chart);
  const robotZ = make('image', { href: 'assets/trajectory-zero.png', x: 427, y: ty(34.60) - 13, width: 49, height: 49, class: 'trajectory-robot' }, chart);
  const roundInput = document.getElementById('trajectory-round');
  const roundOutput = document.getElementById('trajectory-round-value');
  let revealed = 10, selectedRound = 10, animationFrame = 0, animationStarted = false;

  function selectRound(round) {
    selectedRound = Math.min(Math.floor(revealed), Math.max(0, Math.round(round)));
    const d = TRAJECTORY_DATA[selectedRound];
    guide.setAttribute('x1', tx(selectedRound)); guide.setAttribute('x2', tx(selectedRound));
    [[qHead, d.rquest], [zHead, d.rzero]].forEach(([node, value]) => {
      node.setAttribute('cx', tx(selectedRound)); node.setAttribute('cy', ty(value));
    });
    roundInput.value = selectedRound;
    roundOutput.textContent = `Round ${selectedRound}`;
    document.getElementById('trajectory-quest-value').textContent = d.rquest.toFixed(2) + '%';
    document.getElementById('trajectory-zero-value').textContent = d.rzero.toFixed(2) + '%';
    document.getElementById('trajectory-gap-value').textContent = '+' + (d.rquest - d.rzero).toFixed(2) + ' pts';
    chart.setAttribute('aria-label', `Round ${selectedRound}: R-Quest ${d.rquest.toFixed(2)}%, R-Zero ${d.rzero.toFixed(2)}%. Use arrow keys to inspect rounds.`);
  }
  function drawTrajectory(progress) {
    revealed = progress;
    const last = Math.floor(progress), partial = progress - last;
    const q = TRAJECTORY_DATA.slice(0, last + 1).map(d => [tx(d.round), ty(d.rquest)]);
    const z = TRAJECTORY_DATA.slice(0, last + 1).map(d => [tx(d.round), ty(d.rzero)]);
    if (partial > 0 && last < 10) {
      const a = TRAJECTORY_DATA[last], b = TRAJECTORY_DATA[last + 1];
      q.push([tx(progress), ty(a.rquest + (b.rquest - a.rquest) * partial)]);
      z.push([tx(progress), ty(a.rzero + (b.rzero - a.rzero) * partial)]);
    }
    questPath.setAttribute('d', path(q)); zeroPath.setAttribute('d', path(z));
    band.setAttribute('d', path([...q, ...z.slice().reverse()]) + ' Z');
    roundDots.forEach((dot, i) => { dot.q.style.opacity = dot.z.style.opacity = i <= progress ? '1' : '0'; });
    [qEnd, zEnd, robotQ, robotZ].forEach(el => { el.style.opacity = progress >= 10 ? '1' : '0'; });
    selectRound(last);
  }
  function playTrajectory() {
    cancelAnimationFrame(animationFrame);
    if (motion.matches) { drawTrajectory(10); return; }
    drawTrajectory(0);
    const start = performance.now();
    function frame(now) {
      const progress = Math.min(10, (now - start) / 480);
      drawTrajectory(progress);
      if (progress < 10) animationFrame = requestAnimationFrame(frame);
    }
    animationFrame = requestAnimationFrame(frame);
  }
  drawTrajectory(10);
  const trajectoryObserver = new IntersectionObserver(entries => {
    if (!animationStarted && entries.some(entry => entry.isIntersecting)) {
      animationStarted = true; playTrajectory();
    }
  }, { threshold: 0.1 });
  trajectoryObserver.observe(chart);
  document.getElementById('replay-trajectory').addEventListener('click', () => { animationStarted = true; playTrajectory(); });
  function inspectRound(round) { animationStarted = true; cancelAnimationFrame(animationFrame); drawTrajectory(10); selectRound(round); }
  roundInput.addEventListener('input', () => inspectRound(Number(roundInput.value)));
  chart.addEventListener('pointermove', event => {
    const box = chart.getBoundingClientRect();
    selectRound(((event.clientX - box.left) / box.width * 600 - 64) / 43.2);
  });
  chart.addEventListener('click', event => {
    const box = chart.getBoundingClientRect();
    inspectRound(((event.clientX - box.left) / box.width * 600 - 64) / 43.2);
  });
  chart.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowLeft') next = selectedRound - 1;
    if (event.key === 'ArrowRight') next = selectedRound + 1;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = 10;
    if (next !== undefined) { event.preventDefault(); inspectRound(next); }
  });

  // The paper's large-batch approximation; rejection = 1 - pass.
  const probabilityChart = document.getElementById('probability-chart');
  const px = p => 65 + p / 100 * 485;
  const py = probability => 308 - probability / 100 * 264;
  [0, 25, 50, 75, 100].forEach(value => {
    make('line', { x1: 65, x2: 550, y1: py(value), y2: py(value), class: 'chart-grid' }, probabilityChart);
    make('text', { x: 51, y: py(value) + 6, 'text-anchor': 'end', class: 'chart-tick' }, probabilityChart, value);
    make('text', { x: px(value), y: 336, 'text-anchor': 'middle', class: 'chart-tick' }, probabilityChart, value);
  });
  make('text', { x: 307, y: 373, 'text-anchor': 'middle', class: 'chart-axis-title' }, probabilityChart, 'Same-type share p (%)');
  const probabilityAxis = make('text', { transform: 'translate(19 179) rotate(-90)', 'text-anchor': 'middle', class: 'chart-axis-title' }, probabilityChart, 'Pass probability (%)');
  const probabilityArea = make('path', { class: 'probability-area' }, probabilityChart);
  const referenceCurve = make('path', { class: 'probability-reference' }, probabilityChart);
  const selectedCurve = make('path', { class: 'probability-selected' }, probabilityChart);
  const probabilityGuide = make('line', { y1: 44, y2: 308, class: 'chart-guide' }, probabilityChart);
  const referenceDot = make('circle', { r: 5, class: 'probability-reference-dot' }, probabilityChart);
  const selectedDot = make('circle', { r: 6.5, class: 'quest-dot chart-head' }, probabilityChart);
  const share = document.getElementById('type-fraction');
  const budget = document.getElementById('comparisons');
  const tip = document.getElementById('probability-tip');
  let metric = 'pass';
  const pass = (p, k) => Math.pow(1 - p / 100, k) * 100;
  const valueFor = (p, k) => metric === 'pass' ? pass(p, k) : 100 - pass(p, k);
  const referenceK = () => Number(budget.value) === 16 ? 8 : 16;

  function selectShare(p) {
    p = Math.round(Math.min(100, Math.max(0, p)) * 10) / 10;
    const k = Number(budget.value), passing = pass(p, k), rejection = 100 - passing;
    share.value = p;
    document.getElementById('fraction-value').textContent = p.toFixed(p % 1 ? 1 : 0) + '%';
    document.getElementById('pass-value').textContent = passing.toFixed(1) + '%';
    document.getElementById('reject-value').textContent = rejection.toFixed(1) + '%';
    probabilityGuide.setAttribute('x1', px(p)); probabilityGuide.setAttribute('x2', px(p));
    [[selectedDot, k], [referenceDot, referenceK()]].forEach(([dot, comparisons]) => {
      dot.setAttribute('cx', px(p)); dot.setAttribute('cy', py(valueFor(p, comparisons)));
    });
    tip.innerHTML = `<strong>${p.toFixed(p % 1 ? 1 : 0)}% same-type share · K=${k}</strong><span>Pass ${passing.toFixed(1)}% · Reject ${rejection.toFixed(1)}%</span>`;
    document.getElementById('pass-description').textContent = `With K=${k}, a ${p}% same-type share means about ${Math.round(rejection)} in 100 candidates of that type are rejected.`;
    probabilityChart.setAttribute('aria-label', `K=${k}, same-type share ${p}%, pass ${passing.toFixed(1)}%, rejection ${rejection.toFixed(1)}%. Use left and right arrow keys to explore.`);
    document.querySelectorAll('[data-share]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.share) === p)));
  }
  function drawProbability() {
    const k = Number(budget.value), reference = referenceK();
    const samples = Array.from({ length: 201 }, (_, i) => i / 2);
    const selected = samples.map(p => [px(p), py(valueFor(p, k))]);
    selectedCurve.setAttribute('d', path(selected));
    referenceCurve.setAttribute('d', path(samples.map(p => [px(p), py(valueFor(p, reference))])));
    probabilityArea.setAttribute('d', path([[65, 308], ...selected, [550, 308]]) + ' Z');
    probabilityAxis.textContent = metric === 'pass' ? 'Pass probability (%)' : 'Rejection probability (%)';
    document.getElementById('selected-k').textContent = `K=${k}${k === 8 ? ' · paper default' : ''}`;
    document.getElementById('reference-k').textContent = `K=${reference} · comparison`;
    document.getElementById('formula-k').textContent = k;
    document.querySelectorAll('[data-probability-metric]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.probabilityMetric === metric)));
    selectShare(Number(share.value));
    if (!motion.matches && selectedCurve.animate) selectedCurve.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 450 });
  }
  function moveShare(event) {
    const box = probabilityChart.getBoundingClientRect();
    selectShare(((event.clientX - box.left) / box.width * 600 - 65) / 485 * 100);
    tip.hidden = false;
    const outer = document.getElementById('probability-plot').getBoundingClientRect();
    const left = Math.min(Math.max(8, event.clientX - outer.left - 110), Math.max(8, outer.width - 232));
    const top = Math.max(6, (py(valueFor(Number(share.value), Number(budget.value))) / 395 * box.height) - 76);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  probabilityChart.addEventListener('pointermove', moveShare);
  probabilityChart.addEventListener('pointerdown', moveShare);
  probabilityChart.addEventListener('pointerleave', () => { tip.hidden = true; });
  probabilityChart.addEventListener('keydown', event => {
    let p = Number(share.value);
    if (event.key === 'ArrowLeft') p -= 1;
    else if (event.key === 'ArrowRight') p += 1;
    else if (event.key === 'Home') p = 0;
    else if (event.key === 'End') p = 100;
    else return;
    event.preventDefault(); selectShare(p);
  });
  share.addEventListener('input', () => selectShare(Number(share.value)));
  budget.addEventListener('change', drawProbability);
  document.querySelectorAll('[data-probability-metric]').forEach(button => button.addEventListener('click', () => { metric = button.dataset.probabilityMetric; tip.hidden = true; drawProbability(); }));
  document.querySelectorAll('[data-share]').forEach(button => button.addEventListener('click', () => { tip.hidden = true; selectShare(Number(button.dataset.share)); }));
  motion.addEventListener('change', () => { if (motion.matches) { cancelAnimationFrame(animationFrame); drawTrajectory(10); } });
  drawProbability();
})();
