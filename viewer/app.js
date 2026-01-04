// OCR Viewer Application

let entries = [];
let currentIndex = 0;

// DOM Elements
const sourceImage = document.getElementById('sourceImage');
const renderedOutput = document.getElementById('renderedOutput');
const counter = document.getElementById('counter');
const filename = document.getElementById('filename');
const metadata = document.getElementById('metadata');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const jumpInput = document.getElementById('jumpInput');
const jumpBtn = document.getElementById('jumpBtn');

// Initialize
async function init() {
    renderedOutput.innerHTML = '<div class="loading">Loading entries...</div>';

    try {
        const response = await fetch('/api/entries');
        entries = await response.json();

        if (entries.length === 0) {
            renderedOutput.innerHTML = '<div class="loading">No entries found</div>';
            return;
        }

        jumpInput.max = entries.length;
        showEntry(0);
    } catch (error) {
        renderedOutput.innerHTML = `<div class="loading">Error loading entries: ${error.message}</div>`;
    }
}

// Display entry at index
function showEntry(index) {
    if (entries.length === 0) return;

    currentIndex = Math.max(0, Math.min(index, entries.length - 1));
    const entry = entries[currentIndex];

    // Update counter
    counter.textContent = `${currentIndex + 1} / ${entries.length}`;

    // Update filename
    filename.textContent = entry.filename;

    // Update metadata
    let metadataHtml = '';

    // Model info
    const model = entry.model || 'gemini-3-pro-preview';
    metadataHtml += `<span class="metadata-item"><span class="label">Model:</span> <span class="value">${model}</span></span>`;

    // Status
    const status = entry.status || 'unknown';
    const statusClass = status === 'success' ? 'success' : 'error';
    metadataHtml += `<span class="metadata-item"><span class="label">Status:</span> <span class="value ${statusClass}">${status}</span></span>`;

    // Answer length
    const answerLen = entry.answer ? entry.answer.length : 0;
    metadataHtml += `<span class="metadata-item"><span class="label">Output:</span> <span class="value">${answerLen.toLocaleString()} chars</span></span>`;

    metadata.innerHTML = metadataHtml;

    // Update image
    sourceImage.src = `/images/${entry.filename}`;
    sourceImage.alt = entry.filename;

    // Render markdown with HTML support
    const markdown = entry.answer || '';
    renderedOutput.innerHTML = marked.parse(markdown);

    // Update button states
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === entries.length - 1;
}

// Navigation
function prev() {
    showEntry(currentIndex - 1);
}

function next() {
    showEntry(currentIndex + 1);
}

function jumpTo() {
    const num = parseInt(jumpInput.value, 10);
    if (num >= 1 && num <= entries.length) {
        showEntry(num - 1);
        jumpInput.value = '';
    }
}

// Event Listeners
prevBtn.addEventListener('click', prev);
nextBtn.addEventListener('click', next);
jumpBtn.addEventListener('click', jumpTo);

jumpInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') jumpTo();
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    // Don't navigate if typing in input
    if (e.target.tagName === 'INPUT') return;

    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        prev();
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        next();
    } else if (e.key === 'Home') {
        e.preventDefault();
        showEntry(0);
    } else if (e.key === 'End') {
        e.preventDefault();
        showEntry(entries.length - 1);
    }
});

// Start
init();
