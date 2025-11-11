// Global variables
let uploadedFiles = [];
let clusterData = null;
let currentDocCount = 0;
let selectedClusterId = null;

// Setup drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// File upload handler
async function handleFiles(files) {
    const formData = new FormData();
    
    for (let file of files) {
        formData.append('files', file);
    }
    
    // Show progress
    document.getElementById('progressBar').style.display = 'block';
    document.getElementById('progressFill').style.width = '50%';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('uploadStatus').innerHTML = 
            `<p style="color: green;">✅ ${result.uploaded.length} files uploaded successfully</p>`;
        
        // Update stats after a delay
        setTimeout(() => {
            updateStatistics();
            document.getElementById('progressBar').style.display = 'none';
        }, 1000);
        
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML = 
            `<p style="color: red;">❌ Upload failed: ${error.message}</p>`;
    }
}

// Perform clustering
async function performClustering() {
    const statusPill = document.getElementById('clusterStatus');
    const statusText = document.getElementById('clusterStatusText');
    const spinner = statusPill ? statusPill.querySelector('.spinner') : null;
    
    if (currentDocCount === 0) {
        alert('Upload at least one document before generating clusters.');
        return;
    }
    
    try {
        if (statusPill) {
            statusText.textContent = 'Generating clusters…';
            statusPill.hidden = false;
            if (spinner) {
                spinner.style.opacity = '1';
                spinner.style.transform = 'scale(1)';
            }
            statusPill.style.padding = '8px 14px';
        }
        
        const response = await fetch('/api/cluster', {
            method: 'POST'
        });
        
        if (response.ok) {
            clusterData = await response.json();
            selectedClusterId = null;
            displayClusters();
            updateStatistics();
            renderLabelVisualizer();
            renderDocumentPreview();
            if (statusPill) {
                if (spinner) {
                    spinner.style.opacity = '0';
                    spinner.style.transform = 'scale(0.6)';
                }
                statusText.textContent = 'Clusters updated';
                statusPill.style.padding = '8px 18px';
                setTimeout(() => statusPill.hidden = true, 2000);
            }
        } else {
            const error = await response.json();
            alert('Clustering failed: ' + error.error);
            if (statusPill) {
                if (spinner) {
                    spinner.style.opacity = '0';
                    spinner.style.transform = 'scale(0.6)';
                }
                statusPill.hidden = true;
            }
        }
    } catch (error) {
        alert('Error: ' + error.message);
        if (statusPill) {
            if (spinner) {
                spinner.style.opacity = '0';
                spinner.style.transform = 'scale(0.6)';
            }
            statusPill.hidden = true;
        }
    }
}

// Display clusters
function displayClusters() {
    const clusterList = document.getElementById('clusterList');
    
    if (!clusterData || !clusterData.clusters || Object.keys(clusterData.clusters).length === 0) {
        clusterList.innerHTML = `
            <div class="empty-state">
                <p class="empty-icon">🗂️</p>
                <p class="empty-title">No clusters yet</p>
                <p class="empty-text">Upload documents and run clustering to create smart collections automatically.</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    for (let clusterId in clusterData.clusters) {
        const cluster = clusterData.clusters[clusterId];
        const isActive = parseInt(clusterId, 10) === selectedClusterId;
        const topTerms = cluster.top_terms.slice(0, 3).map(t => sanitizeText(t[0])).join(', ');
        html += `
            <div class="cluster-item ${isActive ? 'active' : ''}" onclick="showClusterDetails(${clusterId})">
                <div class="cluster-label">📁 ${sanitizeText(cluster.label)}</div>
                <div class="cluster-info">
                    ${cluster.size} documents • ${topTerms || 'No terms yet'}
                </div>
            </div>
        `;
    }
    
    clusterList.innerHTML = html;
    renderLabelVisualizer();
}

// Show cluster details
async function showClusterDetails(clusterId) {
    try {
        selectedClusterId = clusterId;
        displayClusters();
        const response = await fetch(`/api/cluster/${clusterId}`);
        const cluster = await response.json();
        
        renderLabelVisualizer(cluster);
        renderDocumentPreview(cluster);
        
    } catch (error) {
        alert('Error loading cluster details: ' + error.message);
    }
}

// Close modal
function closeModal() {
    document.getElementById('clusterModal').style.display = 'none';
    document.getElementById('modalContent').innerHTML = '';
}

// Update statistics
async function updateStatistics() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        document.getElementById('totalDocs').textContent = stats.total_documents;
        document.getElementById('totalClusters').textContent = stats.clusters_created;
        document.getElementById('accuracy').textContent = 
            Math.round(stats.parsing_accuracy) + '%';
        document.getElementById('avgTime').textContent = 
            stats.avg_processing_time.toFixed(1) + 's';
        document.getElementById('toolbarDocCount').textContent = stats.total_documents;
        currentDocCount = stats.total_documents;
        
    } catch (error) {
        console.error('Error updating statistics:', error);
    }
}

// Initial load
updateStatistics();
setInterval(updateStatistics, 5000);  // Update every 5 seconds

function renderLabelVisualizer(cluster = null) {
    const container = document.getElementById('labelVisualizer');
    
    if (!cluster && (!clusterData || !clusterData.clusters || Object.keys(clusterData.clusters).length === 0)) {
        container.innerHTML = `
            <div class="empty-state mini">
                <p class="empty-title">No clusters yet</p>
                <p class="empty-text">Run clustering to explore key labels and terms.</p>
            </div>
        `;
        return;
    }
    
    const terms = cluster 
        ? cluster.top_terms
        : Object.values(clusterData.clusters).flatMap(c => c.top_terms.slice(0, 1));
    
    if (!terms.length) {
        container.innerHTML = `
            <div class="empty-state mini">
                <p class="empty-title">No labels available</p>
                <p class="empty-text">Upload more documents to enrich cluster labels.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = terms
        .slice(0, 12)
        .map(term => `<span class="label-chip">${sanitizeText(term[0])}</span>`)
        .join('');
}

function renderDocumentPreview(cluster = null) {
    const panel = document.getElementById('documentPreview');
    
    if (!cluster) {
        panel.innerHTML = `
            <div class="empty-state mini">
                <p class="empty-icon">📄</p>
                <p class="empty-title">Select a cluster</p>
                <p class="empty-text">Choose a smart collection to see representative documents.</p>
            </div>
        `;
        return;
    }
    
    const docs = cluster.documents.slice(0, 4);
    const topTerms = cluster.top_terms.slice(0, 3).map(t => sanitizeText(t[0])).join(', ');
    
    panel.innerHTML = `
        <div class="preview-header">
            <div>
                <p class="section-label">Selected Cluster</p>
                <h3>${sanitizeText(cluster.label)}</h3>
            </div>
            <span class="preview-badge">${cluster.size} docs</span>
        </div>
        <p class="preview-summary">Top terms: ${topTerms || 'N/A'} • Avg length ${Math.round(cluster.average_doc_length)} words</p>
        <div class="preview-list">
            ${docs.map(doc => `
                <div class="doc-preview-item">
                    <div class="doc-preview-header">
                        <span>${sanitizeText(doc.file_name)}</span>
                        <span class="doc-preview-meta">${sanitizeText(doc.format)}</span>
                    </div>
                    <p class="doc-preview-text">${sanitizeText(doc.preview)}</p>
                    <div class="doc-preview-actions">
                        <button class="doc-preview-btn" onclick="openDocument(${doc.id})">View full document</button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function sanitizeText(text = '') {
    if (text === null || text === undefined) {
        return '';
    }
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

async function openDocument(docId) {
    const modal = document.getElementById('clusterModal');
    const modalContent = document.getElementById('modalContent');
    modal.style.display = 'flex';
    modalContent.innerHTML = '<p class="full-doc-loading">Loading document…</p>';
    
    try {
        const response = await fetch(`/api/documents/${docId}`);
        if (!response.ok) {
            throw new Error('Document not found');
        }
        const doc = await response.json();
        const fullText = doc.original_text || doc.clean_text || 'No text available.';
        const tokens = doc.top_unigrams.slice(0, 5).map(t => `<span class="label-chip">${sanitizeText(t[0])}</span>`).join('');
        const formatLabel = doc.format ? doc.format.toUpperCase() : '';
        
        modalContent.innerHTML = `
            <div class="full-doc-header">
                <div>
                    <h2>${sanitizeText(doc.file_name)}</h2>
                    <p class="full-doc-meta">${sanitizeText(formatLabel)} • ${doc.word_count} words</p>
                </div>
                <div class="full-doc-tokens">
                    ${tokens || ''}
                </div>
            </div>
            <div class="full-doc-body">
                ${sanitizeText(fullText).replace(/\n/g, '<br><br>')}
            </div>
        `;
    } catch (error) {
        modalContent.innerHTML = `<p class="full-doc-error">Unable to load document: ${error.message}</p>`;
    }
}
