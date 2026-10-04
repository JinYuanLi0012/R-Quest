'use strict';

// Bar heights and labels share the domain-average values in benchmark-data.js.
// Each domain group uses a non-zero axis; the bars and caption disclose it.
const barPanel = document.getElementById('hero-average-card');
const barColumns = [...barPanel.querySelectorAll('.bar-column')];
const reducedBarMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let barTimers = [];
let barsSeen = false;

function resetBars() {
  barTimers.forEach(clearTimeout);
  barTimers = [];
  barPanel.classList.add('no-animation', 'bars-armed');
  barColumns.forEach(column => column.classList.remove('entered'));
}

function playBars() {
  if (reducedBarMotion.matches) return;
  resetBars();
  // Establish a starting height before enabling the growth transition.
  void barPanel.offsetWidth;
  barPanel.classList.remove('no-animation');
  barPanel.classList.toggle('fast', barsSeen);
  const step = barsSeen ? 75 : 125;
  barColumns.forEach((column, index) => {
    const delay = 30 + (index % 9) * step + Math.floor(index / 9) * 180;
    barTimers.push(setTimeout(() => column.classList.add('entered'), delay));
  });
  barsSeen = true;
}

if (!reducedBarMotion.matches) {
  const barObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !barsSeen) {
        playBars();
      }
    });
  }, { threshold: 0 });
  barObserver.observe(barPanel);
}
document.getElementById('replay-bars').addEventListener('click', playBars);
reducedBarMotion.addEventListener('change', () => {
  if (reducedBarMotion.matches) {
    barTimers.forEach(clearTimeout);
    barPanel.classList.remove('bars-armed', 'no-animation');
    barColumns.forEach(column => column.classList.add('entered'));
  } else playBars();
});
