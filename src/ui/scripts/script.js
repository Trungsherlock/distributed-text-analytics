// Global variables
let uploadedFiles = [];
let clusterData = null;

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
    
    try {
        if (statusPill) {
            statusText.textContent = 'Generating clusters…';
            statusPill.hidden = false;
        }
        
        const response = await fetch('/api/cluster', {
            method: 'POST'
        });
        
        if (response.ok) {
            clusterData = await response.json();
            displayClusters();
            updateStatistics();
            if (statusPill) {
                statusText.textContent = 'Clusters updated';
                setTimeout(() => statusPill.hidden = true, 2000);
            }
        } else {
            const error = await response.json();
            alert('Clustering failed: ' + error.error);
            if (statusPill) statusPill.hidden = true;
        }
    } catch (error) {
        alert('Error: ' + error.message);
        if (statusPill) statusPill.hidden = true;
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
        const topTerms = cluster.top_terms.slice(0, 3).map(t => t[0]).join(', ');
        html += `
            <div class="cluster-item" onclick="showClusterDetails(${clusterId})">
                <div class="cluster-label">📁 ${cluster.label}</div>
                <div class="cluster-info">
                    ${cluster.size} documents • ${topTerms || 'No terms yet'}
                </div>
            </div>
        `;
    }
    
    clusterList.innerHTML = html;
}

// Show cluster details
async function showClusterDetails(clusterId) {
    try {
        const response = await fetch(`/api/cluster/${clusterId}`);
        const cluster = await response.json();
        
        let html = `
            <h2>${cluster.label}</h2>
            <p><strong>Documents:</strong> ${cluster.size}</p>
            <p><strong>Average Length:</strong> ${Math.round(cluster.average_doc_length)} words</p>
            
            <h3>Top Terms</h3>
            <ul>
                ${cluster.top_terms.map(t => 
                    `<li>${t[0]} (score: ${t[1].toFixed(3)})</li>`
                ).join('')}
            </ul>
            
            <h3>Documents in Cluster</h3>
            <div style="max-height: 300px; overflow-y: auto;">
                ${cluster.documents.map(doc => `
                    <div style="background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <strong>${doc.file_name}</strong> (${doc.format})<br>
                        <small>${doc.preview}</small>
                    </div>
                `).join('')}
            </div>
        `;
        
        document.getElementById('modalContent').innerHTML = html;
        document.getElementById('clusterModal').style.display = 'flex';
        
    } catch (error) {
        alert('Error loading cluster details: ' + error.message);
    }
}

// Close modal
function closeModal() {
    document.getElementById('clusterModal').style.display = 'none';
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
        
    } catch (error) {
        console.error('Error updating statistics:', error);
    }
}

// Initial load
updateStatistics();
setInterval(updateStatistics, 5000);  // Update every 5 seconds
