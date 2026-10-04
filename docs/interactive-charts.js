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
  let revealed = 10, selectedRound = 10, animationFrame = 0, chartInView = false;

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
    // Measure frames actually rendered, not time spent loading or in a
    // background tab. A delayed first frame must still begin at round zero.
    let elapsed = 0, previousFrame = null;
    function frame(now) {
      if (document.hidden) { animationFrame = 0; return; }
      if (previousFrame !== null) elapsed += Math.min(80, now - previousFrame);
      previousFrame = now;
      const progress = Math.min(10, elapsed / 600);
      drawTrajectory(progress);
      if (progress < 10) animationFrame = requestAnimationFrame(frame);
    }
    animationFrame = requestAnimationFrame(frame);
  }
  // Start without a click. Like the reference homepage, re-entering the chart
  // restarts the reveal instead of retaining a completed one-shot animation.
  playTrajectory();
  const trajectoryObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.intersectionRatio >= 0.2 && !chartInView) {
        chartInView = true; playTrajectory();
      } else if (entry.intersectionRatio === 0) {
        chartInView = false;
        cancelAnimationFrame(animationFrame);
        drawTrajectory(motion.matches ? 10 : 0);
      }
    });
  }, { threshold: [0, 0.2] });
  trajectoryObserver.observe(chart);
  document.getElementById('replay-trajectory').addEventListener('click', playTrajectory);
  function inspectRound(round) { cancelAnimationFrame(animationFrame); drawTrajectory(10); selectRound(round); }
  document.addEventListener('visibilitychange', () => {
    cancelAnimationFrame(animationFrame);
    if (!document.hidden) playTrajectory();
  });
  window.addEventListener('pageshow', event => { if (event.persisted) playTrajectory(); });
  window.addEventListener('hashchange', () => { if (!location.hash || location.hash === '#top') playTrajectory(); });
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

  // The paper's large-batch approximation, shown only as rejection curves.
  const probabilityChart = document.getElementById('probability-chart');
  const px = p => 65 + p / 100 * 485;
  const py = probability => 248 - probability / 100 * 216;
  [0, 25, 50, 75, 100].forEach(value => {
    make('line', { x1: 65, x2: 550, y1: py(value), y2: py(value), class: 'chart-grid' }, probabilityChart);
    make('text', { x: 51, y: py(value) + 6, 'text-anchor': 'end', class: 'chart-tick' }, probabilityChart, value);
    make('text', { x: px(value), y: 277, 'text-anchor': 'middle', class: 'chart-tick' }, probabilityChart, value);
  });
  make('text', { x: 307, y: 318, 'text-anchor': 'middle', class: 'chart-axis-title' }, probabilityChart, 'Questions sharing the same task (%)');
  make('text', { transform: 'translate(19 140) rotate(-90)', 'text-anchor': 'middle', class: 'chart-axis-title' }, probabilityChart, 'Rejection probability (%)');
  const rejection = (p, k) => (1 - Math.pow(1 - p / 100, k)) * 100;
  const samples = Array.from({ length: 201 }, (_, i) => i / 2);
  const area = make('path', { class: 'probability-area' }, probabilityChart);
  area.setAttribute('d', path([[65, 248], ...samples.map(p => [px(p), py(rejection(p, 8))]), [550, 248]]) + ' Z');
  const curves = [4, 16, 8].map(k => {
    const line = make('path', { class: `rejection-curve budget-${k}`, 'data-k': k }, probabilityChart);
    line.setAttribute('d', path(samples.map(p => [px(p), py(rejection(p, k))])));
    return { k, line };
  });
  const probabilityGuide = make('line', { y1: 32, y2: 248, class: 'chart-guide' }, probabilityChart);
  curves.forEach(curve => { curve.dot = make('circle', { r: curve.k === 8 ? 6 : 4.5, class: `rejection-dot budget-${curve.k}` }, probabilityChart); });
  const tip = document.getElementById('probability-tip');
  let selectedShare = 20;
  function selectShare(p) {
    selectedShare = Math.round(Math.min(100, Math.max(0, p)) * 10) / 10;
    const label = selectedShare.toFixed(selectedShare % 1 ? 1 : 0) + '%';
    document.getElementById('fraction-value').textContent = label;
    probabilityGuide.setAttribute('x1', px(selectedShare)); probabilityGuide.setAttribute('x2', px(selectedShare));
    curves.forEach(({ k, dot }) => {
      const value = rejection(selectedShare, k);
      dot.setAttribute('cx', px(selectedShare)); dot.setAttribute('cy', py(value));
      document.getElementById(`rejection-k${k}`).textContent = value.toFixed(1) + '%';
    });
    tip.innerHTML = `<strong>${label} share the same task</strong>` + [4, 8, 16].map(k => `<span class="tip-budget budget-${k}"><span>K=${k}${k === 8 ? ' · default' : ''}</span><b>${rejection(selectedShare, k).toFixed(1)}%</b></span>`).join('');
    probabilityChart.setAttribute('aria-label', `${label} of questions share the same task. Rejection probabilities: K=4 ${rejection(selectedShare, 4).toFixed(1)}%, K=8 ${rejection(selectedShare, 8).toFixed(1)}%, K=16 ${rejection(selectedShare, 16).toFixed(1)}%. Use arrow keys to explore.`);
  }
  function moveShare(event) {
    const box = probabilityChart.getBoundingClientRect();
    selectShare(((event.clientX - box.left) / box.width * 600 - 65) / 485 * 100);
    tip.hidden = false;
    const outer = document.getElementById('probability-plot').getBoundingClientRect();
    tip.style.left = Math.min(Math.max(8, event.clientX - outer.left - 95), Math.max(8, outer.width - 212)) + 'px';
    tip.style.top = Math.max(6, py(rejection(selectedShare, 8)) / 340 * box.height - 120) + 'px';
  }
  probabilityChart.addEventListener('pointermove', moveShare);
  probabilityChart.addEventListener('pointerdown', moveShare);
  probabilityChart.addEventListener('pointerleave', () => { tip.hidden = true; });
  probabilityChart.addEventListener('keydown', event => {
    let p = selectedShare;
    if (event.key === 'ArrowLeft') p -= 1;
    else if (event.key === 'ArrowRight') p += 1;
    else if (event.key === 'Home') p = 0;
    else if (event.key === 'End') p = 100;
    else return;
    event.preventDefault(); selectShare(p);
  });
  motion.addEventListener('change', () => { if (motion.matches) { cancelAnimationFrame(animationFrame); drawTrajectory(10); } });
  selectShare(20);

})();
