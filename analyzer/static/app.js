document.addEventListener('DOMContentLoaded', () => {
    // UI State
    let analysisData = null;
    let selectedFile = null;
    let activeTab = 'upload';
    
    // Pagination state
    let currentPage = 1;
    const recordsPerPage = 12;
    let filteredRecords = [];
    let selectedRecordIndex = null;
    let redactMode = false; // toggle for PII redaction preview

    // DOM Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const alertContainer = document.getElementById('alert-container');
    
    // Config controls
    const similaritySlider = document.getElementById('similarity_threshold');
    const toxicitySlider = document.getElementById('toxicity_threshold');
    const minTokenSlider = document.getElementById('min_token_threshold');
    const gibberishSlider = document.getElementById('gibberish_threshold');
    const customSchemaCheckbox = document.getElementById('use-custom-schema');
    const schemaTextarea = document.getElementById('schema_json');
    
    const valSimilarity = document.getElementById('val-similarity');
    const valToxicity = document.getElementById('val-toxicity');
    const valMinTokens = document.getElementById('val-min-tokens');
    const valGibberish = document.getElementById('val-gibberish');

    // Ingestion drop zone
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const progressCard = document.getElementById('upload-progress-card');
    const progressFill = document.getElementById('upload-progress-fill');
    const progressFilename = document.getElementById('upload-filename');
    const progressStatus = document.getElementById('upload-status');

    // Dashboard Elements
    const btnDashboardTab = document.getElementById('btn-dashboard-tab');
    const btnRecordsTab = document.getElementById('btn-records-tab');
    const btnReportsTab = document.getElementById('btn-reports-tab');
    const dashFilename = document.getElementById('dash-filename');
    const btnReanalyze = document.getElementById('btn-reanalyze');
    
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    const statTotalRecords = document.getElementById('stat-total-records');
    const statTotalIssues = document.getElementById('stat-total-issues');
    const statIssueProportion = document.getElementById('stat-issue-proportion');
    
    const tokenMin = document.getElementById('token-min');
    const tokenMean = document.getElementById('token-mean');
    const tokenMax = document.getElementById('token-max');
    const dashboardIssuesList = document.getElementById('dashboard-issues-list');

    // Explorer Elements
    const recordSearch = document.getElementById('record-search');
    const issueFilter = document.getElementById('issue-filter');
    const filterIssueCategories = document.getElementById('filter-issue-categories');
    const recordsTbody = document.getElementById('records-tbody');
    const pagination = document.getElementById('pagination');
    const recordDetailPanel = document.getElementById('record-detail-panel');
    const detailEmptyState = document.getElementById('detail-empty-state');
    const detailContent = document.getElementById('detail-content');

    // Reports Elements
    const reportMdPreview = document.getElementById('report-md-preview');
    const reportJsonPreview = document.getElementById('report-json-preview');
    const tabReportBtns = document.querySelectorAll('.tab-report-btn');
    const reportPreviews = document.querySelectorAll('.report-preview');
    const btnCopyMd = document.getElementById('btn-copy-md');
    const btnDownloadMd = document.getElementById('btn-download-md');
    const btnDownloadJson = document.getElementById('btn-download-json');

    /* ==========================================================================
       Tab Navigation
       ========================================================================== */
    function switchTab(tabId) {
        activeTab = tabId;
        navButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabContents.forEach(content => {
            if (content.id === `tab-${tabId}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // Trigger adjustments depending on tab
        if (tabId === 'records') {
            renderExplorer();
        }
    }

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!btn.disabled) {
                switchTab(btn.getAttribute('data-tab'));
            }
        });
    });

    /* ==========================================================================
       Config Controllers
       ========================================================================== */
    similaritySlider.addEventListener('input', (e) => {
        valSimilarity.textContent = parseFloat(e.target.value).toFixed(2);
    });

    toxicitySlider.addEventListener('input', (e) => {
        valToxicity.textContent = parseFloat(e.target.value).toFixed(2);
    });

    minTokenSlider.addEventListener('input', (e) => {
        valMinTokens.textContent = parseInt(e.target.value, 10);
    });

    gibberishSlider.addEventListener('input', (e) => {
        valGibberish.textContent = parseFloat(e.target.value).toFixed(2);
    });

    customSchemaCheckbox.addEventListener('change', (e) => {
        schemaTextarea.disabled = !e.target.checked;
    });

    /* ==========================================================================
       Alerts Utility
       ========================================================================== */
    function showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span>${message}</span>
            <button class="alert-close">&times;</button>
        `;
        alertContainer.appendChild(alert);

        alert.querySelector('.alert-close').addEventListener('click', () => {
            alert.remove();
        });

        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    /* ==========================================================================
       File Ingestion (Drag-and-drop / Browser Selection)
       ========================================================================== */
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFileSelection(files[0]);
        }
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        selectedFile = file;
        progressFilename.textContent = file.name;
        progressCard.style.display = 'block';
        uploadAndAnalyzeFile();
    }

    function uploadAndAnalyzeFile() {
        progressFill.style.width = '10%';
        progressStatus.textContent = 'Uploading dataset...';
        
        // Prep multipart form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('similarity_threshold', similaritySlider.value);
        formData.append('toxicity_threshold', toxicitySlider.value);
        formData.append('min_token_threshold', minTokenSlider.value);
        formData.append('gibberish_threshold', gibberishSlider.value);

        if (customSchemaCheckbox.checked && schemaTextarea.value.trim()) {
            formData.append('schema_json', schemaTextarea.value.trim());
        }

        // Simulate upload progress
        let progress = 10;
        const interval = setInterval(() => {
            if (progress < 85) {
                progress += 5;
                progressFill.style.width = `${progress}%`;
                if (progress === 40) progressStatus.textContent = 'Parsing file rows...';
                if (progress === 65) progressStatus.textContent = 'Running quality detectors...';
                if (progress === 80) progressStatus.textContent = 'Assembling final report...';
            }
        }, 300);

        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(interval);
            progressFill.style.width = '100%';
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP error ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            progressStatus.textContent = 'Complete!';
            showAlert('Dataset successfully analyzed!', 'success');
            
            // Save state globally
            analysisData = data;
            
            // Enable dashboard & records tab
            btnDashboardTab.disabled = false;
            btnRecordsTab.disabled = false;
            btnReportsTab.disabled = false;
            
            setTimeout(() => {
                populateDashboard();
                switchTab('dashboard');
                progressCard.style.display = 'none';
            }, 500);
        })
        .catch(err => {
            clearInterval(interval);
            progressCard.style.display = 'none';
            showAlert(`Analysis failed: ${err.message}`, 'error');
            console.error(err);
        });
    }

    btnReanalyze.addEventListener('click', () => {
        if (selectedFile) {
            progressFilename.textContent = selectedFile.name;
            progressCard.style.display = 'block';
            switchTab('upload');
            uploadAndAnalyzeFile();
        }
    });

    /* ==========================================================================
       Dashboard Populator
       ========================================================================== */
    function populateDashboard() {
        if (!analysisData) return;
        
        dashFilename.textContent = analysisData.filename;
        
        // Metrics
        const qualityPercentage = Math.round(analysisData.metrics.quality_score * 100);
        gaugeValue.textContent = `${qualityPercentage}%`;
        
        // Animating circular SVG gauge: stroke-dasharray is "circumferencePercent, 100"
        gaugeFill.setAttribute('stroke-dasharray', `${qualityPercentage}, 100`);
        
        // Gauge color mapping
        if (qualityPercentage >= 80) {
            gaugeFill.style.stroke = 'var(--success)';
        } else if (qualityPercentage >= 50) {
            gaugeFill.style.stroke = 'var(--warning)';
        } else {
            gaugeFill.style.stroke = 'var(--error)';
        }

        statTotalRecords.textContent = Number(analysisData.summary.total_records).toLocaleString();
        statTotalIssues.textContent = Number(analysisData.summary.total_issues).toLocaleString();
        statIssueProportion.textContent = `${(analysisData.metrics.issue_record_proportion * 100).toFixed(1)}%`;
        
        tokenMin.textContent = analysisData.metrics.min_tokens;
        tokenMean.textContent = analysisData.metrics.mean_tokens;
        tokenMax.textContent = analysisData.metrics.max_tokens;

        // Categories list
        dashboardIssuesList.innerHTML = '';
        const selectFilterCategories = document.getElementById('filter-issue-categories');
        selectFilterCategories.innerHTML = '';

        // Categories details
        Object.keys(analysisData.category_counts).forEach(cat => {
            const count = analysisData.category_counts[cat];
            
            // Format nice label
            const displayLabel = cat.replace(/_/g, ' ');

            // Build dashboard widgets
            const widget = document.createElement('div');
            widget.className = 'issue-item-widget';
            widget.innerHTML = `
                <div class="issue-info">
                    <span class="issue-label">${displayLabel}</span>
                    <span class="issue-count">${count}</span>
                </div>
                <span class="issue-badge-count issue-badg-${cat}">${count > 0 ? 'Action Needed' : 'Clean'}</span>
            `;
            widget.addEventListener('click', () => {
                issueFilter.value = cat;
                applyFilterAndSearch();
                switchTab('records');
            });
            dashboardIssuesList.appendChild(widget);

            // Populate Record Explorer dropdown filter categories
            const option = document.createElement('option');
            option.value = cat;
            option.textContent = `${displayLabel} (${count})`;
            selectFilterCategories.appendChild(option);
        });

        // Previews
        reportMdPreview.textContent = analysisData.report_md;
        reportJsonPreview.textContent = analysisData.report_json;
    }

    /* ==========================================================================
       Record Explorer & Pagination
       ========================================================================== */
    function renderExplorer() {
        if (!analysisData) return;
        applyFilterAndSearch();
    }

    function applyFilterAndSearch() {
        const query = recordSearch.value.toLowerCase().trim();
        const filterVal = issueFilter.value;
        currentPage = 1;
        redactMode = false; // reset redact toggler when switching records
        
        filteredRecords = analysisData.records.filter(record => {
            // Text search over fields
            let matchesSearch = true;
            if (query !== '') {
                matchesSearch = Object.values(record.fields).some(val => {
                    if (val === null) return false;
                    return String(val).toLowerCase().includes(query);
                });
            }

            // Issue type filters
            let matchesFilter = true;
            if (filterVal === 'issues') {
                matchesFilter = record.issues.length > 0;
            } else if (filterVal === 'clean') {
                matchesFilter = record.issues.length === 0;
            } else if (filterVal !== 'all') {
                matchesFilter = record.issues.some(iss => iss.category === filterVal);
            }

            return matchesSearch && matchesFilter;
        });

        renderRecordsList();
    }

    // Dynamic filtering triggers
    recordSearch.addEventListener('input', applyFilterAndSearch);
    issueFilter.addEventListener('change', applyFilterAndSearch);

    function renderRecordsList() {
        recordsTbody.innerHTML = '';
        
        const startIndex = (currentPage - 1) * recordsPerPage;
        const endIndex = Math.min(startIndex + recordsPerPage, filteredRecords.length);
        const pageItems = filteredRecords.slice(startIndex, endIndex);

        if (pageItems.length === 0) {
            recordsTbody.innerHTML = `
                <tr>
                    <td colspan="4" style="text-align: center; color: var(--text-dark); padding: 40px;">
                        No records match current filter criteria.
                    </td>
                </tr>
            `;
            renderPagination(0);
            return;
        }

        pageItems.forEach(rec => {
            const tr = document.createElement('tr');
            if (selectedRecordIndex === rec.index) {
                tr.className = 'selected';
            }

            // Build preview text
            let preview = '';
            // Get first string field or concatenate values
            const textFields = Object.entries(rec.fields)
                .filter(([_, v]) => typeof v === 'string')
                .map(([_, v]) => v);
            
            if (textFields.length > 0) {
                preview = textFields[0];
            } else {
                preview = JSON.stringify(rec.fields);
            }
            if (preview.length > 80) preview = preview.substring(0, 80) + '...';

            // Location coordinates
            let locText = `Row ${rec.index + 1}`;
            if (rec.location) {
                if (rec.location.line_number !== null) locText = `Line ${rec.location.line_number}`;
                else if (rec.location.array_index !== null) locText = `Index ${rec.location.array_index}`;
                else if (rec.location.row_index !== null) locText = `Group ${rec.location.row_group}, Row ${rec.location.row_index}`;
            }

            // Badges
            let badgesHtml = '';
            if (rec.issues.length > 0) {
                // Get unique categories
                const categories = [...new Set(rec.issues.map(i => i.category))];
                categories.forEach(cat => {
                    badgesHtml += `<span class="mini-badge issue-badg-${cat}">${cat.replace(/_/g, ' ')}</span>`;
                });
            } else {
                badgesHtml = '<span class="mini-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">Clean</span>';
            }

            tr.innerHTML = `
                <td style="font-weight:600; width: 60px;">${rec.index}</td>
                <td title="${escapeHtml(preview)}">${escapeHtml(preview)}</td>
                <td class="loc-lbl" style="width: 120px;">${locText}</td>
                <td style="width: 180px;"><div class="findings-badges">${badgesHtml}</div></td>
            `;

            tr.addEventListener('click', () => {
                // Remove selected class from sibling rows
                document.querySelectorAll('#records-tbody tr').forEach(row => row.classList.remove('selected'));
                tr.classList.add('selected');
                selectRecord(rec.index);
            });

            recordsTbody.appendChild(tr);
        });

        renderPagination(filteredRecords.length);
    }

    function renderPagination(totalItems) {
        pagination.innerHTML = '';
        const totalPages = Math.ceil(totalItems / recordsPerPage);
        
        if (totalPages <= 1) {
            pagination.innerHTML = `<span>Showing ${totalItems} records</span>`;
            return;
        }

        const startSpan = (currentPage - 1) * recordsPerPage + 1;
        const endSpan = Math.min(currentPage * recordsPerPage, totalItems);

        const textSpan = document.createElement('span');
        textSpan.textContent = `Showing ${startSpan}-${endSpan} of ${totalItems} records`;
        pagination.appendChild(textSpan);

        const btnGroup = document.createElement('div');
        btnGroup.className = 'pagination-btn-group';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'page-btn';
        prevBtn.disabled = currentPage === 1;
        prevBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
        `;
        prevBtn.addEventListener('click', () => {
            currentPage--;
            renderRecordsList();
        });
        btnGroup.appendChild(prevBtn);

        const nextBtn = document.createElement('button');
        nextBtn.className = 'page-btn';
        nextBtn.disabled = currentPage === totalPages;
        nextBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
        `;
        nextBtn.addEventListener('click', () => {
            currentPage++;
            renderRecordsList();
        });
        btnGroup.appendChild(nextBtn);

        pagination.appendChild(btnGroup);
    }

    /* ==========================================================================
       Record Details Inspector & Redactions
       ========================================================================== */
    function selectRecord(index) {
        selectedRecordIndex = index;
        const rec = analysisData.records.find(r => r.index === index);
        if (!rec) return;

        detailEmptyState.style.display = 'none';
        detailContent.style.display = 'block';
        
        // Location coordinates label
        let locText = `Record Index ${rec.index}`;
        if (rec.location) {
            if (rec.location.line_number !== null) locText = `Line Number: ${rec.location.line_number}`;
            else if (rec.location.array_index !== null) locText = `Array Element: ${rec.location.array_index}`;
            else if (rec.location.row_index !== null) locText = `Row Group: ${rec.location.row_group}, Row: ${rec.location.row_index}`;
        }

        // Check if there is PII to redact
        const hasPII = rec.issues.some(i => i.category === 'pii');

        // Render Fields box
        let fieldsHtml = '';
        Object.entries(rec.fields).forEach(([fieldName, value]) => {
            // Apply text formatting
            let displayVal = value;
            if (value === null) {
                displayVal = '<span style="color: var(--text-dark); font-style: italic;">null</span>';
            } else if (typeof value === 'object') {
                displayVal = JSON.stringify(value);
            } else {
                displayVal = escapeHtml(String(value));
            }

            // Highlighting rules (only if original mode and there are issues with spans in this field)
            const fieldIssues = rec.issues.filter(i => i.field_name === fieldName);
            if (!redactMode && fieldIssues.length > 0 && typeof value === 'string') {
                displayVal = applyHighlights(value, fieldIssues);
            } else if (redactMode && hasPII && rec.redacted_fields && rec.redacted_fields[fieldName] !== undefined) {
                // Show redacted fields
                displayVal = escapeHtml(String(rec.redacted_fields[fieldName]));
            }

            // Headers for fields
            let fieldHeader = `<div class="field-name">${fieldName}</div>`;
            if (fieldName === 'text' && hasPII && rec.redacted_fields) {
                fieldHeader = `
                    <div class="redact-header">
                        <div class="field-name">${fieldName}</div>
                        <button class="redact-toggle-btn" id="btn-toggle-redact">
                            ${redactMode ? 'Show Original' : 'Preview Redacted'}
                        </button>
                    </div>
                `;
            }

            fieldsHtml += `
                <div class="field-block">
                    ${fieldHeader}
                    <div class="field-val">${displayVal}</div>
                </div>
            `;
        });

        // Render Issues details
        let issuesHtml = '';
        if (rec.issues.length > 0) {
            rec.issues.forEach(iss => {
                const badgeClass = `issue-badg-${iss.category}`;
                const label = iss.category.replace(/_/g, ' ');
                
                let extraDetail = '';
                if (iss.pii_category) extraDetail = `<br><strong>PII Category:</strong> ${iss.pii_category}`;
                if (iss.score !== null) extraDetail = `<br><strong>Score:</strong> ${parseFloat(iss.score).toFixed(3)}`;
                
                issuesHtml += `
                    <div class="detail-issue-card iss-${iss.category}">
                        <h5>
                            <span style="text-transform: capitalize;">${label}</span>
                            <span class="mini-badge ${badgeClass}">Alert</span>
                        </h5>
                        <p>${escapeHtml(iss.detail)}${extraDetail}</p>
                    </div>
                `;
            });
        } else {
            issuesHtml = `
                <div class="detail-issue-card" style="background: rgba(16, 185, 129, 0.03); border: 1px solid rgba(16, 185, 129, 0.15);">
                    <h5 style="color: var(--success);">Clean Record</h5>
                    <p>No quality issues, toxicity, duplicates, or formatting violations detected in this training example.</p>
                </div>
            `;
        }

        detailContent.innerHTML = `
            <div class="detail-sec">
                <h4>Data Coordinates</h4>
                <div class="loc-pill-detail">${locText}</div>
            </div>
            
            <div class="detail-sec">
                <h4>Record Content</h4>
                <div class="record-fields-box">
                    ${fieldsHtml}
                </div>
            </div>

            <div class="detail-sec">
                <h4>Quality Findings</h4>
                <div class="issues-scroller">
                    ${issuesHtml}
                </div>
            </div>
        `;

        // Wire up toggle button if it exists
        const btnToggleRedact = document.getElementById('btn-toggle-redact');
        if (btnToggleRedact) {
            btnToggleRedact.addEventListener('click', () => {
                redactMode = !redactMode;
                selectRecord(index); // re-render detail view with updated toggle
            });
        }
    }

    // Highlight spans in field text
    function applyHighlights(text, issues) {
        // Collect spans that have start and end coordinates
        const spansToHighlight = [];
        issues.forEach(iss => {
            if (iss.span && iss.span.start !== undefined && iss.span.end !== undefined) {
                spansToHighlight.push({
                    start: iss.span.start,
                    end: iss.span.end,
                    category: iss.category
                });
            }
        });

        if (spansToHighlight.length === 0) return escapeHtml(text);

        // Sort spans right-to-left to avoid coordinate shifting while injecting html tags
        spansToHighlight.sort((a, b) => b.start - a.start);

        // Resolve overlaps: remove nested or overlapping spans
        const resolvedSpans = [];
        let lastStart = Infinity;
        for (const span of spansToHighlight) {
            if (span.end <= lastStart) {
                resolvedSpans.push(span);
                lastStart = span.start;
            }
        }

        let result = text;
        resolvedSpans.forEach(span => {
            const prefix = result.substring(0, span.start);
            const substring = result.substring(span.start, span.end);
            const suffix = result.substring(span.end);
            
            const highlightClass = span.category === 'pii' ? 'highlight-pii' : 'highlight-toxicity';
            
            result = prefix + `<span class="${highlightClass}">${escapeHtml(substring)}</span>` + suffix;
        });

        return result;
    }

    /* ==========================================================================
       Reports Copy & Download
       ========================================================================== */
    tabReportBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const reportType = btn.getAttribute('data-report-type');
            
            tabReportBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            reportPreviews.forEach(p => {
                if (p.id === `report-${reportType}-preview`) {
                    p.classList.add('active');
                } else {
                    p.classList.remove('active');
                }
            });
        });
    });

    btnCopyMd.addEventListener('click', () => {
        if (!analysisData) return;
        navigator.clipboard.writeText(analysisData.report_md)
            .then(() => showAlert('Markdown report copied to clipboard!', 'info'))
            .catch(() => showAlert('Failed to copy report.', 'error'));
    });

    btnDownloadMd.addEventListener('click', () => {
        if (!analysisData) return;
        downloadBlob(analysisData.report_md, 'quality_report.md', 'text/markdown');
    });

    btnDownloadJson.addEventListener('click', () => {
        if (!analysisData) return;
        downloadBlob(analysisData.report_json, 'quality_report.json', 'application/json');
    });

    function downloadBlob(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    /* ==========================================================================
       Helpers
       ========================================================================== */
    function escapeHtml(text) {
        if (typeof text !== 'string') return text;
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
