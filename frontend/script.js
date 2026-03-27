const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const analysisSection = document.getElementById('analysis-section');
const statusText = document.getElementById('status-text');
const results = document.getElementById('results');
const loadingSpinner = document.getElementById('loading-spinner');
const chartCard = document.getElementById('chart-card');
let audioChart;

// Setup Chart.js
const ctx = document.getElementById('audioChart').getContext('2d');
audioChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array.from({length: 20}, (_, i) => `${i*50}ms`),
        datasets: [] // will be filled after prediction
    },
    options: {
        responsive: true,
        plugins: { legend: { display: true, labels: {color: '#94a3b8'} } },
        scales: {
            y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' } },
            x: { ticks: { color: '#94a3b8' } }
        }
    }
});

function handleFile(file) {
    if(!file) return;
    
    analysisSection.classList.remove('hidden');
    chartCard.classList.remove('hidden');
    setTimeout(() => {
        analysisSection.classList.remove('opacity-0');
        chartCard.classList.remove('opacity-0');
    }, 50);

    results.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    statusText.innerText = `Analyzing "${file.name}"...`;
    
    // clear previous panels and chart
    document.getElementById('model-panels').innerHTML = '';
    audioChart.data.datasets = [];
    audioChart.update();
    
    // Mock Chart animation while loading (empty datasets)
    let counter = 0;
    const interval = setInterval(() => {
        audioChart.data.datasets.forEach(ds => {
            ds.data = Array.from({length: 20}, () => Math.random() * 50);
        });
        audioChart.update();
        counter++;
        if(counter > 15) clearInterval(interval);
    }, 100);

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        clearInterval(interval);
        loadingSpinner.classList.add('hidden');
        results.classList.remove('hidden');
        
        if (data.error) {
            statusText.innerText = "Error analyzing file.";
            console.error(data.error);
            return;
        }

        statusText.innerText = "Analysis Complete";
        
        // helper for styling
        function createPanel(name, info, color) {
            const isFake = info.prediction === 'FAKE';
            const barColor = isFake ? 'from-rose-500 to-rose-400' : 'from-emerald-400 to-emerald-500';
            return `
                <div class="">
                    <div class="flex justify-between items-end mb-2">
                        <span class="text-sm text-slate-400 uppercase tracking-widest font-semibold">${name}</span>
                        <span class="text-3xl font-bold tracking-wider ${isFake ? 'text-rose-500' : 'text-emerald-500'}">${info.prediction}</span>
                    </div>
                    <div class="w-full bg-slate-700 h-4 rounded-full overflow-hidden mb-6 mt-4">
                        <div class="prob-bar h-full bg-gradient-to-r transition-all duration-1000 ease-out ${barColor}" style="width:0%"></div>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span class="text-emerald-400">REAL: ${info.real_probability}%</span>
                        <span class="text-rose-400">FAKE: ${info.fake_probability}%</span>
                    </div>
                </div>`;
        }

        // prepare colors for chart lines
        const palette = ['#3b82f6','#f43f5e','#fde047','#10b981','#8b5cf6'];
        let idx = 0;
        const lineData = [];

        // iterate through models in response
        for (const [modelName, info] of Object.entries(data)) {
            const displayName = modelName.charAt(0).toUpperCase() + modelName.slice(1);
            const panelHtml = createPanel(displayName, info);
            document.getElementById('model-panels').insertAdjacentHTML('beforeend', panelHtml);

            // generate line data
            const base = info.fake_probability; // use fake % as center
            const fake = info.fake_probability;
            const real = info.real_probability;
            const isFake = info.prediction === 'FAKE';
            function genLine(baseval) {
                return Array.from({length:20}, () => Math.max(0, Math.min(100, baseval + (Math.random()*10 -5))));
            }
            // convert hex to rgba with alpha 0.1
            const hex = palette[idx % palette.length];
            const rgb = hex.match(/[0-9a-fA-F]{2}/g).map(h => parseInt(h,16)).join(',');
            lineData.push({
                label: displayName,
                data: genLine(isFake ? fake : real),
                borderColor: hex,
                backgroundColor: `rgba(${rgb},0.1)`,
                fill: true,
                tension: 0.4
            });
            idx++;
        }

        // animate bars after panels added
        document.querySelectorAll('.prob-bar').forEach((bar,i) => {
            const modelInfo = Object.values(data)[i];
            const isFake = modelInfo.prediction === 'FAKE';
            const width = isFake ? modelInfo.fake_probability : modelInfo.real_probability;
            setTimeout(() => { bar.style.width = width + '%'; }, 50);
        });

        // update chart with new datasets
        audioChart.data.datasets = lineData;
        audioChart.update();
    })
    .catch(err => {
        clearInterval(interval);
        statusText.innerText = "Connection Error. Is the backend server running?";
        loadingSpinner.classList.add('hidden');
    });
}

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

dropZone.addEventListener('drop', (e) => {
    handleFile(e.dataTransfer.files[0]);
});
