const BACKEND_HEALTH_URL = 'https://junsiang26-odiocheck-backend.hf.space/health';

// Wake up the backend immediately when the page loads
fetch(BACKEND_HEALTH_URL).catch(() => console.log('Backend waking up...'));

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const analysisSection = document.getElementById('analysis-section');
const statusText = document.getElementById('status-text');
const results = document.getElementById('results');
const loadingSpinner = document.getElementById('loading-spinner');
const chartCard = document.getElementById('chart-card');

// -------------------------------------------------------
// Chart setup
// -------------------------------------------------------
const ctx = document.getElementById('audioChart').getContext('2d');
let audioChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
        responsive: true,
        animation: { duration: 600, easing: 'easeInOutQuart' },
        plugins: {
            legend: { display: true, labels: { color: '#94a3b8', font: { size: 12 } } },
            tooltip: {
                callbacks: {
                    label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}% fake`,
                    title: items => `Segment @ ${items[0].label}s`
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: { color: '#94a3b8', callback: v => v + '%' },
                grid: { color: 'rgba(148,163,184,0.1)' },
                title: { display: true, text: 'Fake Probability (%)', color: '#64748b' }
            },
            x: {
                ticks: {
                    color: '#94a3b8', callback: (_, i, ticks) => {
                        // Show fewer labels when there are many windows
                        const step = Math.max(1, Math.floor(ticks.length / 8));
                        return i % step === 0 ? audioChart.data.labels[i] + 's' : '';
                    }
                },
                grid: { color: 'rgba(148,163,184,0.05)' },
                title: { display: true, text: 'Time (seconds)', color: '#64748b' }
            }
        }
    }
});

// Palette and display names for the four models
const MODEL_META = {
    wav2vec2: { label: 'Wav2Vec2', color: '#3b82f6' },
    aasist: { label: 'AASIST', color: '#f43f5e' },
    cqcc_baseline: { label: 'CQCC Baseline', color: '#fbbf24' },
    custom_hybrid: { label: 'Proposed Custom Hybrid', color: '#10b981' },
};

// -------------------------------------------------------
// File handling
// -------------------------------------------------------
function handleFile(file) {
    if (!file) return;

    // Show sections
    analysisSection.classList.remove('hidden');
    chartCard.classList.remove('hidden');
    setTimeout(() => {
        analysisSection.classList.remove('opacity-0');
        chartCard.classList.remove('opacity-0');
    }, 50);

    results.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    statusText.innerText = `Analyzing "${file.name}"…`;

    // Clear previous state
    document.getElementById('model-panels').innerHTML = '';
    audioChart.data.labels = [];
    audioChart.data.datasets = [];
    audioChart.update();

    // Animated placeholder while waiting: a single pulsing dataset
    const placeholder = {
        label: 'Analyzing…',
        data: Array.from({ length: 20 }, (_, i) => 45 + Math.sin(i / 2) * 10),
        borderColor: 'rgba(99,102,241,0.5)',
        backgroundColor: 'rgba(99,102,241,0.05)',
        borderDash: [4, 4],
        fill: true,
        tension: 0.4,
        pointRadius: 0,
    };
    audioChart.data.labels = Array.from({ length: 20 }, (_, i) => i);
    audioChart.data.datasets = [placeholder];
    audioChart.update();

    let tick = 0;
    const loadingAnim = setInterval(() => {
        tick++;
        placeholder.data = Array.from({ length: 20 }, (_, i) =>
            45 + Math.sin((i + tick) / 2) * 10
        );
        audioChart.update('none'); // skip animation for perf
    }, 80);

    const formData = new FormData();
    formData.append('file', file);

    const HF_API_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost'
        ? '/api/predict' 
        : 'https://junsiang26-odiocheck-backend.hf.space/api/predict';
    
    fetch(HF_API_URL, { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            clearInterval(loadingAnim);
            loadingSpinner.classList.add('hidden');

            if (data.error) {
                statusText.innerText = 'Error analyzing file.';
                console.error(data.error);
                return;
            }

            renderResults(data);
        })
        .catch(() => {
            clearInterval(loadingAnim);
            loadingSpinner.classList.add('hidden');
            statusText.innerText = 'Connection error. Is the backend running?';
        });
}

// -------------------------------------------------------
// Render results from the new response shape:
//   data.overall   → { model: { prediction, fake_probability, real_probability } }
//   data.timeline  → { model: [fake_prob_pct, ...] }
//   data.window_labels → [centre_sec, ...]
// -------------------------------------------------------
function renderResults(data) {
    const { overall, timeline, window_labels } = data;

    statusText.innerText = 'Analysis Complete';
    results.classList.remove('hidden');

    // --- Model panels (overall verdict) ---
    const panelsEl = document.getElementById('model-panels');
    panelsEl.innerHTML = '';

    for (const [key, info] of Object.entries(overall)) {
        const meta = MODEL_META[key] || { label: key, color: '#94a3b8' };
        const isFake = info.prediction === 'FAKE';
        const barColor = isFake ? 'from-rose-500 to-rose-400' : 'from-emerald-400 to-emerald-500';
        const displayPct = isFake ? info.fake_probability : info.real_probability;

        panelsEl.insertAdjacentHTML('beforeend', `
            <div>
                <div class="flex justify-between items-end mb-2">
                    <span class="text-sm text-slate-400 uppercase tracking-widest font-semibold"
                          style="color:${meta.color}">${meta.label}</span>
                    <span class="text-3xl font-bold tracking-wider ${isFake ? 'text-rose-500' : 'text-emerald-500'}">
                        ${info.prediction}
                    </span>
                </div>
                <div class="text-xs text-slate-500 mb-2">
                    Fake: <span class="text-slate-300">${info.fake_probability}%</span>
                    &nbsp;·&nbsp;
                    Real: <span class="text-slate-300">${info.real_probability}%</span>
                </div>
                <div class="w-full bg-slate-700 h-4 rounded-full overflow-hidden mb-6 mt-1">
                    <div class="prob-bar h-full bg-gradient-to-r transition-all duration-1000 ease-out ${barColor}"
                         style="width:0%"
                         data-width="${displayPct}">
                    </div>
                </div>
            </div>`);
    }

    // Animate bars
    requestAnimationFrame(() => {
        document.querySelectorAll('.prob-bar').forEach(bar => {
            bar.style.width = bar.dataset.width + '%';
        });
    });

    // --- Timeline chart (real data) ---
    // window_labels are now start-of-segment times (0, 2, 4 ...)
    // For short audio with a single window, we pad with the audio-end label
    // so the chart shows a line rather than a lonely dot.
    let labels = [...window_labels];
    let timelineValues = {};
    Object.entries(timeline).forEach(([k, v]) => { timelineValues[k] = [...v]; });

    if (labels.length === 1) {
        // Estimate audio duration: single window = TARGET_LEN / 16000 ≈ 4.025s
        const audioEnd = parseFloat((labels[0] + 4.025).toFixed(2));
        labels.push(audioEnd);
        Object.keys(timelineValues).forEach(k => timelineValues[k].push(timelineValues[k][0]));
    }

    audioChart.data.labels = labels;
    audioChart.data.datasets = Object.entries(timelineValues).map(([key, values]) => {
        const meta = MODEL_META[key] || { label: key, color: '#94a3b8' };
        const hex = meta.color;
        const rgb = hex.match(/[0-9a-fA-F]{2}/g).map(h => parseInt(h, 16)).join(',');
        return {
            label: meta.label,
            data: values,
            borderColor: hex,
            backgroundColor: `rgba(${rgb},0.08)`,
            fill: true,
            tension: 0.4,
            pointRadius: values.length <= 20 ? 4 : 2,
            pointHoverRadius: 6,
        };
    });

    // Add a 50% threshold reference line
    audioChart.data.datasets.push({
        label: 'Decision threshold (50%)',
        data: Array(labels.length).fill(50),
        borderColor: 'rgba(255,255,255,0.2)',
        borderDash: [6, 4],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0,
    });

    audioChart.update();
}

// -------------------------------------------------------
// Drop zone wiring
// -------------------------------------------------------
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, e => { e.preventDefault(); e.stopPropagation(); });
});
dropZone.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));