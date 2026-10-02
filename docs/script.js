'use strict';

const themeButton = document.getElementById('theme-toggle');
const themeLabel = document.getElementById('theme-label');
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const next = theme === 'dark' ? 'Light' : 'Dark';
  themeLabel.textContent = next;
  themeButton.setAttribute('aria-label', `Switch to ${next.toLowerCase()} theme`);
  try { localStorage.setItem('rquest-theme', theme); } catch (_) {}
}
try { setTheme(localStorage.getItem('rquest-theme') || 'light'); } catch (_) { setTheme('light'); }
themeButton.addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

const problemTabs = [...document.querySelectorAll('[data-tab-group="problem"]')];
function chooseProblem(tab) {
  problemTabs.forEach(button => {
    const active = button === tab;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
    document.getElementById(button.getAttribute('aria-controls')).hidden = !active;
  });
}
problemTabs.forEach(tab => tab.addEventListener('click', () => chooseProblem(tab)));
function keyboardTabs(buttons, select) {
  buttons.forEach((button, index) => button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
    if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = buttons.length - 1;
    if (next === undefined) return;
    event.preventDefault(); select(buttons[next]); buttons[next].focus();
  }));
}
keyboardTabs(problemTabs, chooseProblem);

const fraction = document.getElementById('type-fraction');
const comparisons = document.getElementById('comparisons');
function updateProbability() {
  const p = Number(fraction.value), k = Number(comparisons.value);
  const percent = Math.pow(1 - p / 100, k) * 100;
  document.getElementById('fraction-value').textContent = p + '%';
  document.getElementById('pass-value').textContent = percent.toFixed(1) + '%';
  document.getElementById('pass-bar').style.width = percent + '%';
  document.getElementById('pass-description').textContent = `At a ${p}% same-type fraction, about ${Math.round(percent)} in 100 candidates pass with K = ${k}.`;
}
fraction.addEventListener('input', updateProbability);
comparisons.addEventListener('change', updateProbability);
updateProbability();

let benchmarkBackbone = 'qwen';
let benchmarkDomain = 'math';
const domainLabels = { math:'Mathematical reasoning', code:'Code generation', general:'General-domain reasoning' };
function renderBenchmarks() {
  const model = BENCHMARKS[benchmarkBackbone];
  const data = model.domains[benchmarkDomain];
  const table = document.getElementById('benchmark-table');
  table.dataset.domain = benchmarkDomain;
  table.querySelector('caption').textContent = domainLabels[benchmarkDomain] + ' benchmark scores for ' + model.name;
  table.querySelector('thead').innerHTML = '<tr><th scope="col">Method</th>' + data.headers.map((name,index) => '<th scope="col"' + (index===0?' class="average-column"':'') + '>' + name + '</th>').join('') + '</tr>';
  const best = data.headers.map((_,index) => Math.max(...data.rows.map(row => row[index+1])));
  table.querySelector('tbody').innerHTML = data.rows.map(row => '<tr' + (row[0]==='R-Quest'?' class="ours"':'') + '><th scope="row">' + row[0] + (row[0]==='R-Quest'?' <span>Ours</span>':'') + '</th>' + row.slice(1).map((score,index) => '<td class="' + (index===0?'average-column ':'') + (score===best[index]?'best':'') + '">' + score.toFixed(2) + '</td>').join('') + '</tr>').join('');
  document.querySelectorAll('[data-backbone]').forEach(button => { const active=button.dataset.backbone===benchmarkBackbone;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active)); });
  document.querySelectorAll('[data-benchmark-domain]').forEach(button => { const active=button.dataset.benchmarkDomain===benchmarkDomain;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active)); });
  document.getElementById('benchmark-metric').textContent = benchmarkDomain==='math' ? 'AMC and AIME: mean@32. Other math benchmarks: greedy-decoding accuracy.' : benchmarkDomain==='code' ? 'HumanEval+ and MBPP+: pass@1 on the full EvalPlus test suites.' : 'SuperGPQA, MMLU-Pro and BBEH: exact-match accuracy under greedy decoding.';
}
document.querySelectorAll('[data-backbone]').forEach(button => button.addEventListener('click', () => {benchmarkBackbone=button.dataset.backbone;renderBenchmarks();}));
document.querySelectorAll('[data-benchmark-domain]').forEach(button => button.addEventListener('click', () => {benchmarkDomain=button.dataset.benchmarkDomain;renderBenchmarks();}));
renderBenchmarks();

const evidence = {
  ablation:{asset:'ablation',index:'01 / COMPONENT ABLATIONS',title:'Both signals matter.<br>Validity sustains the gains.',body:'Either validity or novelty feedback alone achieves a higher peak score than R-Zero. However, validity feedback is more critical for sustained gains: its removal leads to late-stage collapse, whereas the variant without novelty feedback shows a milder decline and remains above the base model.',caption:'Average performance on seven benchmarks over ten rounds on Qwen3-4B-Base.',alt:'Ablation comparison: removing validity feedback leads to late-stage collapse; without novelty the decline is milder.'},
  validity:{asset:'question-validity',index:'02 / QUESTION VALIDITY',title:'More valid questions,<br>before filtering.',body:'R-Quest increases the valid-question rate from 88.0% in round one to 94.5–97.5% throughout rounds four to ten. In contrast, the rate for R-Zero falls from 76.5% to 49.0%. This indicates that R-Quest steers question generation toward valid problems as self-evolution proceeds.',caption:'Question validity before filtering. Each method and round is evaluated on 200 sampled questions; shaded bands show pointwise 95% Wilson confidence intervals.',alt:'R-Quest retains 94.5% question validity at round ten, compared with R-Zero’s 49.0%.'},
  answers:{asset:'answer-quality',index:'03 / ANSWER QUALITY',title:'More reliable<br>self-generated supervision.',body:'The two methods begin with comparable majority-answer accuracy on valid questions, but diverge sharply over subsequent rounds. By round ten, R-Quest retains 54.69% accuracy, whereas R-Zero falls to just 2.06%. The deterioration in R-Zero extends from question generation to answer supervision.',caption:'Majority-answer accuracy on valid questions, measured against independently verified reference answers.',alt:'At round ten R-Quest retains 54.69% majority-answer accuracy on valid questions, versus 2.06% for R-Zero.'},
  diversity:{asset:'diversity',index:'04 / TASK DIVERSITY',title:'Less repetition.<br>A balanced novelty gate.',body:'The top-five question-type share for R-Zero rises from 17.0% to 69.0% between rounds one and five. R-Quest with the default K = 8 reaches 24.0% in round five while maintaining higher performance. Increasing K to 16 further suppresses repetition, but can limit mathematical gains.',caption:'Question-type clusters (bars, left axis) and mathematical performance (lines, right axis). Labels above the bars give the top-five group proportion.',alt:'R-Quest keeps recurring mathematical task types less concentrated than R-Zero. K=8 balances task diversity and performance.'}
};
const evidenceTabs=[...document.querySelectorAll('[data-evidence]')];
function chooseEvidence(button) {
  const d=evidence[button.dataset.evidence];
  evidenceTabs.forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
  document.getElementById('evidence-panel').setAttribute('aria-labelledby',button.id);
  const img=document.getElementById('evidence-image');img.src=`assets/${d.asset}.webp`;img.alt=d.alt;
  document.getElementById('evidence-index').textContent=d.index;
  document.getElementById('evidence-title').innerHTML=d.title;
  document.getElementById('evidence-body').textContent=d.body;
  document.getElementById('evidence-caption').textContent=d.caption;
  document.getElementById('evidence-pdf').href=`assets/${d.asset}.pdf`;
  const zoom=document.getElementById('evidence-zoom');zoom.dataset.zoom=img.src;zoom.dataset.caption=d.caption;
}
evidenceTabs.forEach(b=>b.addEventListener('click',()=>chooseEvidence(b)));
keyboardTabs(evidenceTabs,chooseEvidence);

const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  const img=document.getElementById('dialog-image');img.src=button.dataset.zoom;img.alt=button.querySelector('img').alt;
  document.getElementById('dialog-caption').textContent=button.dataset.caption;
  dialog.showModal();document.body.style.overflow='hidden';
}));
document.getElementById('close-figure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
dialog.addEventListener('close',()=>{document.body.style.overflow='';});

document.getElementById('copy-citation').addEventListener('click',async()=>{
  const text=document.getElementById('bibtex').textContent;
  const status=document.getElementById('copy-status');
  try {
    if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text);}
    else{const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok)throw new Error('Clipboard unavailable');}
    status.textContent='Citation copied.';
    document.getElementById('copy-citation').textContent='Copied';
    setTimeout(()=>{document.getElementById('copy-citation').textContent='Copy citation';},2500);
  }catch(_){status.textContent='Select the BibTeX text above and copy it manually.';}
});

const navLinks=[...document.querySelectorAll('.header nav a')];
const observer=new IntersectionObserver(entries=>{
  entries.forEach(entry=>{if(entry.isIntersecting)navLinks.forEach(link=>link.classList.toggle('active',link.hash==='#'+entry.target.id));});
},{rootMargin:'-15% 0px -55% 0px'});
['problem','method','results','cite'].forEach(id=>observer.observe(document.getElementById(id)));
