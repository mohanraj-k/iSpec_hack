/**
 * MDD Automation Utility - Frontend JavaScript
 */

class MDDAutomationApp {
    constructor() {
        this.isProcessing = false;
        this.isDatabaseInitialized = false;
        this.init();
    }

    getMaxUploadMB() {
        try {
            const meta = document.querySelector('meta[name="max-upload-mb"]');
            const val = meta ? parseInt(meta.getAttribute('content'), 10) : NaN;
            return Number.isFinite(val) && val > 0 ? val : 50;
        } catch (_) {
            return 50;
        }
    }

    init() {
        this.bindEvents();
        this.checkDatabaseStatus();
    }

    bindEvents() {
        // Database summary button
        document.getElementById('summaryBtn').addEventListener('click', () => {
            this.showDatabaseSummary();
        });



        // Database rebuild - open modal instead
        document.getElementById('rebuildDbBtn').addEventListener('click', () => {
            const rebuildModal = new bootstrap.Modal(document.getElementById('rebuildModal'));
            rebuildModal.show();
        });

        // Rebuild Check_logic_MDD button (runs standalone aggregator)
        const rebuildCheckLogicBtn = document.getElementById('rebuildCheckLogicBtn');
        if (rebuildCheckLogicBtn) {
            rebuildCheckLogicBtn.addEventListener('click', async () => {
                try {
                    rebuildCheckLogicBtn.disabled = true;
                    rebuildCheckLogicBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Rebuilding...';
                    const resp = await fetch('/rebuild-check-logic-mdd', { method: 'POST' });
                    const result = await resp.json();
                    if (result.success) {
                        this.showAlert(`Check_logic MDD rebuilt successfully. Rows: ${result.rows}`, 'success');
                    } else {
                        this.showAlert('Failed to rebuild Check_logic MDD: ' + (result.error || 'Unknown error'), 'danger');
                    }
                } catch (err) {
                    this.showAlert('Error: ' + err.message, 'danger');
                } finally {
                    rebuildCheckLogicBtn.disabled = false;
                    rebuildCheckLogicBtn.innerHTML = '<i class="fas fa-table me-2"></i>Rebuild Check_logic_MDD';
                }
            });
        }

        // Modal Upload & Rebuild button
        document.getElementById('modalUploadBtn').addEventListener('click', async () => {
            await this.handleModalUploadAndRebuild();
        });

        // Help documentation
        document.getElementById('helpBtn').addEventListener('click', () => {
            this.showHelp();
        });

        // File upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });

        

        // File input change
        document.getElementById('targetFile').addEventListener('change', (e) => {
            this.validateFile(e.target.files[0]);
        });

        // Table filtering - will be bound after results are loaded
        this.currentResults = [];
    }

    bindFilterEvents() {
        const filterRadios = document.querySelectorAll('input[name="resultFilter"]');
        filterRadios.forEach(radio => {
            radio.addEventListener('change', () => {
                this.filterResults(radio.value);
            });
        });
        
        // Quality slider filter
        const qualitySlider = document.getElementById('qualitySlider');
        if (qualitySlider) {
            qualitySlider.addEventListener('input', (e) => {
                this.filterByQuality(parseInt(e.target.value));
            });
            qualitySlider.addEventListener('change', (e) => {
                this.filterByQuality(parseInt(e.target.value));
            });
        }
    }

    filterResults(filterType) {
        if (!this.currentResults || this.currentResults.length === 0) return;

        let filteredResults = this.currentResults;
        
        if (filterType === 'matches') {
            filteredResults = this.currentResults.filter(result => {
                const matchType = result['Match Type'] || '';
                return matchType !== 'No Match' && matchType !== 'Error' && matchType !== '';
            });
        }

        // Apply quality filter if slider exists
        const qualitySlider = document.getElementById('qualitySlider');
        if (qualitySlider) {
            const qualityLevel = parseInt(qualitySlider.value);
            filteredResults = this.applyQualityFilter(filteredResults, qualityLevel);
        }

        // Update statistics based on filtered results
        this.updateStatistics(filteredResults);

        this.populateMatchResults(filteredResults);
    }

    filterByQuality(qualityLevel) {
        if (!this.currentResults || this.currentResults.length === 0) return;

        console.log('filterByQuality called with level:', qualityLevel);

        // Update filter level badge
        this.updateFilterLevelBadge(qualityLevel);

        // Get current radio filter state
        const selectedRadio = document.querySelector('input[name="resultFilter"]:checked');
        const filterType = selectedRadio ? selectedRadio.value : 'all';

        // Apply both radio filter and quality filter
        let filteredResults = this.currentResults;
        
        if (filterType === 'matches') {
            filteredResults = this.currentResults.filter(result => {
                const matchType = result['Match Type'] || '';
                return matchType !== 'No Match' && matchType !== 'Error' && matchType !== '';
            });
        }

        filteredResults = this.applyQualityFilter(filteredResults, qualityLevel);
        
        // Update statistics based on filtered results
        this.updateStatistics(filteredResults);
        
        // Update table display
        this.populateMatchResults(filteredResults);
    }

    applyQualityFilter(results, qualityLevel) {
        const qualityFilters = {
            1: ['Excellent Match'], // Excellent only
            2: ['Excellent Match', 'Good Match'], // Good & above
            3: ['Excellent Match', 'Good Match', 'Moderate Match'], // Moderate & above  
            4: ['Excellent Match', 'Good Match', 'Moderate Match', 'Weak Match'] // All matches
        };

        const allowedTypes = qualityFilters[qualityLevel] || qualityFilters[1];
        
        console.log('Quality filter level:', qualityLevel);
        console.log('Allowed match types:', allowedTypes);
        console.log('Total results before filter:', results.length);
        
        const filteredResults = results.filter(result => {
            const matchType = result['Match Type'] || '';
            const isAllowed = allowedTypes.includes(matchType);
            if (!isAllowed) {
                console.log('Filtering out match type:', matchType);
            }
            return isAllowed;
        });
        
        console.log('Results after quality filter:', filteredResults.length);
        return filteredResults;
    }

    updateFilterLevelBadge(qualityLevel) {
        // Badge element has been removed - this function is now a no-op
        // The quality level is indicated by the slider position itself
        return;
    }

    updateStatistics(filteredResults) {
        // Calculate statistics for filtered results aligned to slider categories
        let excellentMatches = 0;
        let goodMatches = 0;
        let moderateMatches = 0;
        let weakMatches = 0; // includes legacy 'Low Match'
        let noMatches = 0;   // includes errors/unknowns

        filteredResults.forEach(result => {
            const matchType = result['Match Type'] || '';
            if (matchType === 'Excellent Match') {
                excellentMatches++;
            } else if (matchType === 'Good Match') {
                goodMatches++;
            } else if (matchType === 'Moderate Match') {
                moderateMatches++;
            } else if (matchType === 'Weak Match' || matchType === 'Low Match') {
                weakMatches++;
            } else {
                noMatches++;
            }
        });

        // Update the statistics display
        const elExcellent = document.getElementById('excellentMatches');
        const elGood = document.getElementById('goodMatches');
        const elModerate = document.getElementById('moderateMatches');
        const elWeak = document.getElementById('weakMatches');
        const elNo = document.getElementById('noMatches');
        if (elExcellent) elExcellent.textContent = excellentMatches;
        if (elGood) elGood.textContent = goodMatches;
        if (elModerate) elModerate.textContent = moderateMatches;
        if (elWeak) elWeak.textContent = weakMatches;
        if (elNo) elNo.textContent = noMatches;

        console.log('Statistics updated:', { excellentMatches, goodMatches, moderateMatches, weakMatches, noMatches });
    }



    validateFile(file) {
        const uploadBtn = document.getElementById('uploadBtn');
        
        if (!file) {
            uploadBtn.disabled = true;
            return false;
        }

        // Check file type/extension (.xlsx or .csv)
        const nameLower = (file.name || '').toLowerCase();
        const isXlsx = nameLower.endsWith('.xlsx');
        const isCsv = nameLower.endsWith('.csv');
        if (!isXlsx && !isCsv) {
            this.showAlert('Please select a valid file (.xlsx or .csv)', 'danger');
            uploadBtn.disabled = true;
            return false;
        }

        // Check file size (limit from server config)
        const limitMB = this.getMaxUploadMB();
        const maxSize = limitMB * 1024 * 1024;
        if (file.size > maxSize) {
            this.showAlert(`File size must be less than ${limitMB}MB`, 'danger');
            uploadBtn.disabled = true;
            return false;
        }

        uploadBtn.disabled = false;
        return true;
    }

    async handleFileUpload() {
        if (this.isProcessing) {
            console.log('[handleFileUpload] Already processing, aborting new upload.');
            return;
        }
        this.isProcessing = true;

        const fileInput = document.getElementById('targetFile');
        const file = fileInput.files[0];
        if (!file) {
            this.showAlert('Please select a file to upload.', 'danger');
            this.isProcessing = false;
            return;
        }

        // Get sponsor name from input
        const sponsorInput = document.getElementById('sponsorName');
        const sponsorName = sponsorInput ? sponsorInput.value : '';
        // Get match scope from radio buttons
        const scopeEl = document.querySelector('input[name="match_scope"]:checked');
        const matchScope = scopeEl ? scopeEl.value : 'across';

        // Generate a unique upload_id client-side so we can start polling immediately
        const uploadId = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : Date.now().toString(36);
        const formData = new FormData();
        formData.append('target_file', file);
        formData.append('upload_id', uploadId);
        formData.append('sponsor_name', sponsorName);
        formData.append('match_scope', matchScope);

        // Show the progress section and start polling right away
        this.showProgressSection();
        console.log('[handleFileUpload] Progress section shown, starting pollProgress');
        this.pollProgress(uploadId);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('[handleFileUpload] Upload response:', data);
            if (data.error) {
                this.updateProgress(100, 'Error: ' + data.error);
                this.showAlert(data.error, 'danger');
                setTimeout(() => this.hideProgressSection(), 2000);
                this.isProcessing = false;
                return;
            }
            const uploadId = data.upload_id;
            if (!uploadId) {
                this.updateProgress(100, 'Processing completed.');
                setTimeout(() => this.hideProgressSection(), 1000);
                if (data.result) this.showResults(data.result);
                this.isProcessing = false;
                return;
            }
            // Start polling for progress
            this.pollProgress(uploadId, data.result);
        })
        .catch(error => {
            this.updateProgress(100, 'Upload failed.');
            this.showAlert('Upload failed: ' + error.message, 'danger');
            setTimeout(() => this.hideProgressSection(), 2000);
            this.isProcessing = false;
        });
    }


    // Update circular progress UI
    updateRebuildProgress(percent, textMsg = '') {
        const container = document.getElementById('rebuildProgressContainer');
        const circle = document.getElementById('rebuildProgressCircle');
        const percentLabel = document.getElementById('rebuildProgressPercent');
        const textLabel = document.getElementById('rebuildProgressText');
        if (!container || !circle) return;
        container.classList.remove('d-none');
        const deg = Math.floor((percent / 100) * 360);
        circle.style.background = `conic-gradient(#0d6efd ${deg}deg, #e9ecef ${deg}deg)`;
        percentLabel.textContent = `${percent}%`;
        if (textMsg) textLabel.textContent = textMsg;
    }

    // Handle modal upload + rebuild flow
    async handleModalUploadAndRebuild() {
        const modalUploadBtn = document.getElementById('modalUploadBtn');
        const fileInput = document.getElementById('modalDbFiles');
        const files = fileInput.files;
        const originalText = modalUploadBtn.innerHTML;
        modalUploadBtn.disabled = true;
        modalUploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
        try {
            if (files && files.length > 0) {
                // Total size check for multi-file upload based on server limit
                const limitMB = this.getMaxUploadMB();
                const totalSize = Array.from(files).reduce((sum, f) => sum + (f.size || 0), 0);
                if (totalSize > limitMB * 1024 * 1024) {
                    this.showAlert(`Total size of selected files exceeds ${limitMB}MB`, 'danger');
                    modalUploadBtn.disabled = false;
                    modalUploadBtn.innerHTML = originalText;
                    return;
                }
                const formData = new FormData();
                for (const file of files) {
                    if (!this.validateFile(file)) {
                        modalUploadBtn.disabled = false;
                        modalUploadBtn.innerHTML = originalText;
                        return;
                    }
                    formData.append('db_files', file);
                }
                const uploadResp = await fetch('/upload-mdd', { method: 'POST', body: formData });
                const uploadResult = await uploadResp.json();
                if (!uploadResult.success) {
                    this.showAlert('Upload failed: ' + (uploadResult.error || 'Unknown error'), 'danger');
                    modalUploadBtn.disabled = false;
                    modalUploadBtn.innerHTML = originalText;
                    return;
                }
                this.showAlert(uploadResult.message, 'success');
            }
            // Start progress simulation
            this.updateRebuildProgress(10, 'Uploading completed');
            // Hide file section
            document.getElementById('modalFileSection').classList.add('d-none');
            // Start rebuild in background with server-side progress
            const startResp = await fetch('/rebuild-database-start', { method: 'POST' });
            const startData = await startResp.json();
            if (!startResp.ok || !startData.success) {
                const msg = startData && startData.error ? startData.error : 'Failed to start rebuild';
                this.showAlert(msg, 'danger');
            } else {
                const rebuildId = startData.rebuild_id;
                // Poll server for rebuild progress
                const pollInterval = setInterval(async () => {
                    try {
                        const r = await fetch(`/progress/${rebuildId}`);
                        const data = await r.json();
                        const pct = typeof data.percent === 'number' ? data.percent : parseInt(data.percent) || 0;
                        const msg = data.message || '';
                        this.updateRebuildProgress(pct, msg);
                        if (pct >= 100) {
                            clearInterval(pollInterval);
                            this.updateRebuildProgress(100, 'Rebuild completed');
                            try {
                                await this.checkDatabaseStatus();
                            } catch (e) {
                                console.warn('checkDatabaseStatus failed:', e);
                            }
                            // Hide modal after short delay on completion
                            setTimeout(() => {
                                const modalEl = document.getElementById('rebuildModal');
                                const bsModal = bootstrap.Modal.getInstance(modalEl);
                                if (bsModal) bsModal.hide();
                                // reset UI
                                document.getElementById('modalFileSection').classList.remove('d-none');
                                document.getElementById('rebuildProgressContainer').classList.add('d-none');
                            }, 1200);
                        }
                    } catch (e) {
                        console.warn('Progress polling error:', e);
                    }
                }, 1000);
            }

            // Modal is hidden upon completion in the polling block above
        } catch (error) {
            this.showAlert('Error: ' + error.message, 'danger');
        } finally {
            modalUploadBtn.disabled = false;
            modalUploadBtn.innerHTML = originalText;
            // Reset file input
            fileInput.value = '';
        }
    }

    pollProgress(uploadId, finalResult = null) {
        console.log(`[pollProgress] Polling for uploadId=${uploadId}`);
        fetch(`/progress/${uploadId}`)
            .then(response => response.json())
            .then(data => {
                console.log('[pollProgress] Received data:', data);
                this.updateProgress(data.percent, data.message);
                if (data.percent < 100) {
                    setTimeout(() => this.pollProgress(uploadId, finalResult), 1000);
                } else {
                    setTimeout(() => this.hideProgressSection(), 1000);
                    if (finalResult) {
                        this.showResults(finalResult);
                    }
                }
            })
            .catch(() => {
                setTimeout(() => this.pollProgress(uploadId, finalResult), 1000);
            });
    }

    showProgressSection() {
        const progressSection = document.getElementById('progressSection');
        const resultsSection = document.getElementById('resultsSection');
        
        progressSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        console.log('[showProgressSection] progressSection should now be visible');
    }

    hideProgressSection() {
        const progressSection = document.getElementById('progressSection');
        progressSection.classList.add('d-none');
        this.isProcessing = false;
        console.log('[hideProgressSection] progressSection hidden, isProcessing set to false');
    }

    updateProgress(percentage, message) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        console.log(`[updateProgress] Called with percentage=${percentage}, message=${message}`);
        
        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `${percentage}%`;
        progressText.innerHTML = `<i class=\"fas fa-cogs me-2\"></i>${message}`;
    }

    showResults(result) {
        this.hideProgressSection();
        console.log('Rs-Line291 showResults',typeof result);
        console.log('Rs-Line292 showResults',result);
        const resultsSection = document.getElementById('resultsSection');
        const stats = result.statistics;
        
        // Update statistics (use detailed breakdown aligned with slider categories)
        const elExcellent = document.getElementById('excellentMatches');
        const elGood = document.getElementById('goodMatches');
        const elModerate = document.getElementById('moderateMatches');
        const elWeak = document.getElementById('weakMatches');
        const elNo = document.getElementById('noMatches');

        if (elExcellent) elExcellent.textContent = (stats && (stats.excellent_matches || 0)) || 0;
        if (elGood) elGood.textContent = (stats && (stats.good_matches || 0)) || 0;
        if (elModerate) elModerate.textContent = (stats && (stats.moderate_matches || 0)) || 0;
        // Backend key is low_matches; keep weak_matches as a fallback if present
        if (elWeak) elWeak.textContent = (stats && ((stats.low_matches ?? stats.weak_matches) || 0)) || 0;
        if (elNo) elNo.textContent = (stats && (stats.no_matches || 0)) || 0;

        
        // Update download links with robust error handling
        if (result.download_urls) {
            try {
                console.log('Download URLs received:', result.download_urls);
                
                // Wait for DOM to be ready before accessing elements
                setTimeout(() => {
                    try {
                        const downloadCsv = document.getElementById('downloadCsv');
                        const downloadJson = document.getElementById('downloadJson');
                        
                        console.log('Download elements found:', {
                            csv: !!downloadCsv,
                            json: !!downloadJson
                        });
                        
                        // Enable CSV download
                        if (downloadCsv && result.download_urls.csv) {
                            downloadCsv.href = result.download_urls.csv;
                            downloadCsv.classList.remove('disabled');
                            downloadCsv.removeAttribute('aria-disabled');
                            console.log('CSV download enabled:', result.download_urls.csv);
                        }
                        
                        // Enable JSON download (prefer matches_only)
                        if (downloadJson) {
                            // const jsonUrl = result.download_urls.matches_json || result.download_urls.json;
                            const jsonUrl = result.download_urls.json;
                            if (jsonUrl) {
                                downloadJson.href = jsonUrl;
                                downloadJson.classList.remove('disabled');
                                downloadJson.removeAttribute('aria-disabled');
                                console.log('JSON download enabled:', jsonUrl);
                            }
                        }
                    } catch (domError) {
                        console.error('DOM access error:', domError);
                    }
                }, 100);
            } catch (error) {
                console.error('Error updating download links:', error);
                console.error('Error stack:', error.stack);
            }
        }
        
        // Store results and populate match results table
        this.currentResults = result.results || [];
        this.populateMatchResults(this.currentResults);
        
        // Bind filter events
        this.bindFilterEvents();
        
        // Initialize quality slider to "Excellent only" by default
        setTimeout(() => {
            const qualitySlider = document.getElementById('qualitySlider');
            if (qualitySlider) {
                qualitySlider.value = 1; // Default to Excellent only
                this.updateFilterLevelBadge(1);
                console.log('Initializing quality filter to level 1');
                this.filterByQuality(1); // Apply default filter
            } else {
                console.error('Quality slider not found during initialization');
            }
        }, 100);
        
        resultsSection.classList.remove('d-none');
        
        // Show success message
        this.showAlert(
            `Processing completed successfully! ${stats.total_rows} rows processed with ${stats.match_rate}% match rate.`,
            'success'
        );
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    populateMatchResults(results) {
        const tbody = document.getElementById('matchResultsBody');
        tbody.innerHTML = ''; // Clear existing results
        
        // Display all results - even low similarity matches are valuable for reference
        if (!results || results.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="10" class="text-center text-muted py-4">
                    <i class="fas fa-info-circle me-2"></i>
                    No results found. Please check the file format and try again.
                    <br><small>Ensure the file contains valid MDD data with description fields.</small>
                </td>
            `;
            tbody.appendChild(row);
            return;
        }
        
        // Sort results by confidence score in descending order
        const sortedResults = [...results].sort((a, b) => {
            const confidenceA = parseFloat(a['Confidence Score'] || 0);
            const confidenceB = parseFloat(b['Confidence Score'] || 0);
            return confidenceB - confidenceA; // Descending order
        });
        
        sortedResults.forEach((result, index) => {
            const row = document.createElement('tr');
            
            // Get match type badge class with green/amber color scheme
            const matchType = result['Match Type'] || 'Unknown';
            let badgeClass = 'bg-secondary';
            
            // Green variations for good matches - using darker green
            if (matchType === 'Excellent Match') badgeClass = 'bg-success';
            else if (matchType === 'Good Match') badgeClass = 'bg-success';
            // Amber variations for moderate/weak matches
            else if (matchType === 'Moderate Match') badgeClass = 'bg-warning';
            else if (matchType === 'Low Match' || matchType === 'Weak Match') badgeClass = 'bg-warning';
            // Red for no match
            else if (matchType === 'No Match') badgeClass = 'bg-danger';
            
            // Legacy support for old classifications
            else if (matchType === 'Complete') badgeClass = 'bg-success';
            else if (matchType === 'Partial') badgeClass = 'bg-warning';
            
            // Get sponsor and study from Reference fields
            const sponsor = result['Reference Sponsor'] || 
                          (result['Origin Study'] && result['Origin Study'].includes('_') ? 
                           result['Origin Study'].split('_')[0] : 
                           result['Origin Study'] || 'Unknown');
            
            const study = result['Reference Study'] || 
                         (result['Origin Study'] && result['Origin Study'].includes('_') ? 
                          result['Origin Study'].split('_')[1] : 
                          'Unknown');
            
            // Get target data from uploaded file (using correct field names)
            const targetCheckName = result['DQ Name'] || result['Check Name'] || 'N/A';
            const targetCheckDesc = result['Target Check Description'] || result['DQ Description'] || result['Edit Check Description'] || 'N/A';
            const targetQueryText = result['Target Query Text'] || result['Standard Query text'] || result['Query Text'] || 'N/A';
            
            // Get reference data from matched database record
            const refCheckName = result['Reference Check Name'] || result['Reference DQ Name'] || 'N/A';
            const refCheckDesc = result['Reference Check Description'] || 'N/A';
            const refQueryText = result['Reference Query Text'] || result['Reference Query'] || 'N/A';
            const refPseudoCode = result['Check logic'] ||result['Reference Pseudo Code'] || result['Pseudo Code'] || 'N/A';
            
            // Get Is Match Found flag
            const isMatchFound = result['Is Match Found'] || (matchType === 'Complete' ? 'YES' : 'NO');
            
            // Format confidence as raw FAISS score (0-1) with 3 decimal places
            const confidence = parseFloat(result['Confidence Score'] || 0).toFixed(3);
            
            // Get match explanation
            const matchExplanation = result['match_explanation'] || result['Match Explanation'] || result['match_reason'] || 'No explanation available';
            
            // Get enhanced metadata fields
            const domain = result['Domain'] || 'N/A';
            const formName = result['Form Name'] || 'N/A';
            const visitName = result['Visit Name'] || 'N/A';
            const relationalDomains = result['Relational Domains'] || 'N/A';
            const primaryVars = result['Primary Domain Variables (Pre-Conf)'] || 'N/A';
            const relationalVars = result['Relational Domain Variables'] || 'N/A';
            const relationalDynamicVars = result['Relational Dynamic Variables'] || 'N/A';
            const dynamicVars = result['Dynamic Panel Variables (Pre-Conf)'] || 'N/A';
            const queryTarget = result['Query Target (Pre-Conf)'] || 'N/A';
            const originStudy = result['Origin Study (Copy Source Study)'] || 'N/A';
            const techCode = result['Pseudo Tech Code (Copy Source Study)'] || 'N/A';
            const operationalNotes = result['Operational Notes'] || '';
            // to be continued
            row.innerHTML = `
                <td class="text-center">${index + 1}</td>
                <td class="text-center">
                    <span class="badge ${isMatchFound === 'YES' ? 'bg-success' : 'bg-danger'}">${isMatchFound}</span>
                </td>
                <td class="text-center">
                    <span class="badge ${badgeClass}">${matchType}</span>
                </td>
                <td class="text-center">
                    <strong>${confidence}</strong>
                </td>
                <td>
                    <div class="text-center">
                        <span class="text-primary fw-bold d-block">${sponsor}</span>
                        <small class="text-muted">${study}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; max-height: 60px; overflow-y: auto;">
                        <small>${this.escapeHtml(refCheckName)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 80px; overflow-y: auto;">
                        <small>${this.escapeHtml(refCheckDesc)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 80px; overflow-y: auto;">
                        <small>${this.escapeHtml(refQueryText)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; max-height: 60px; overflow-y: auto;">
                        <small>${this.escapeHtml(refPseudoCode)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; max-height: 60px; overflow-y: auto;">
                        <small>${this.escapeHtml(targetCheckName)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 80px; overflow-y: auto;">
                        <small>${this.escapeHtml(targetCheckDesc)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 80px; overflow-y: auto;">
                        <small>${this.escapeHtml(targetQueryText)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal;">
                        <small class="text-dark fw-bold">${this.escapeHtml(domain)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3;">
                        <small class="text-dark">${this.escapeHtml(formName)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3;">
                        <small class="text-dark">${this.escapeHtml(visitName)}</small>
                    </div>
                </td>
                
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-dark">${this.escapeHtml(primaryVars)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-dark">${this.escapeHtml(dynamicVars)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-dark">${this.escapeHtml(relationalDomains)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-dark">${this.escapeHtml(relationalVars)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-dark">${this.escapeHtml(relationalDynamicVars)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3;">
                        <small class="text-dark">${this.escapeHtml(queryTarget)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3;">
                        <small class="text-dark">${this.escapeHtml(originStudy)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: pre-wrap; line-height: 1.3; max-height: 100px; overflow-y: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px; padding: 4px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px;">
                        <pre class="mb-0" style="white-space: pre-wrap;">${this.escapeHtml(techCode)}</pre>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.3; max-height: 70px; overflow-y: auto;">
                        <small class="text-danger fw-bold">${this.escapeHtml(operationalNotes)}</small>
                    </div>
                </td>
                <td>
                    <div style="word-wrap: break-word; white-space: normal; line-height: 1.4; max-height: 100px; overflow-y: auto; font-size: 13px; padding: 4px;">
                        <span class="text-dark">${this.formatMatchExplanation(matchExplanation)}</span>
                    </div>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    }

    truncateText(text, maxLength) {
        if (!text || text === 'N/A') return text;
        if (text.length <= maxLength) return this.escapeHtml(text);
        return this.escapeHtml(text.substring(0, maxLength)) + '...';
    }

    formatMatchExplanation(explanation) {
        if (!explanation || explanation === 'No explanation available') {
            return '<em class="text-muted">No explanation available</em>';
        }
        
        // Split explanation into parts and format with proper line breaks
        let formatted = this.escapeHtml(explanation)
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-primary">$1</strong>') // Bold headers
            .replace(/✓/g, '<span class="text-success">✓</span>') // Green checkmarks
            .replace(/⚠/g, '<span class="text-warning">⚠</span>') // Warning symbols
            .replace(/Hybrid Score:/g, '<br><strong>Hybrid Score:</strong>')
            .replace(/Reference:/g, '<br><strong>Reference:</strong>')
            .replace(/Recommendation:/g, '<br><strong class="text-info">Recommendation:</strong>');
        
        return formatted;
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.getElementById('statusAlert');
        const messageSpan = document.getElementById('statusMessage');
        
        alertDiv.className = `alert alert-${type}`;
        messageSpan.textContent = message;
        
        // Remove spinner for non-info alerts
        const spinner = alertDiv.querySelector('.spinner-border');
        if (type !== 'info' && spinner) {
            spinner.style.display = 'none';
        } else if (type === 'info' && spinner) {
            spinner.style.display = 'inline-block';
        }
        
        alertDiv.classList.remove('d-none');
        
        // Auto-hide non-error alerts after 5 seconds
        if (type !== 'danger') {
            setTimeout(() => {
                alertDiv.classList.add('d-none');
            }, 5000);
        }
    }

    hideAlert() {
        const alertDiv = document.getElementById('statusAlert');
        alertDiv.classList.add('d-none');
    }

    async showDatabaseSummary() {
        const summaryBtn = document.getElementById('summaryBtn');
        const summaryDiv = document.getElementById('dbSummary');
        const contentDiv = document.getElementById('summaryContent');

        // Toggle collapse
        const bsCollapse = new bootstrap.Collapse(summaryDiv, { toggle: false });
        
        // If already shown, just toggle
        if (summaryDiv.classList.contains('show')) {
            bsCollapse.toggle();
            summaryBtn.innerHTML = '<i class="fas fa-chart-pie me-2"></i>View Database Summary';
            return;
        }

        try {
            // Update button state
            summaryBtn.disabled = true;
            summaryBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';

            const response = await fetch('/database-summary');
            const result = await response.json();

            if (result.error) {
                this.showAlert(result.error, 'danger');
                return;
            }

            // Create summary content with clean table layout
            let summaryHTML = `
                <div class="row mb-3">
                    <div class="col-md-4">
                        <h6 class="text-info mb-2">Database Statistics</h6>
                        <table class="table table-sm table-borderless">
                            <tbody>
                                <tr><td><strong>Total Records:</strong></td><td>${Object.values(result.file_breakdown || {}).reduce((a, b) => a + b, 0)}</td></tr>
                                <tr><td><strong>Total Files:</strong></td><td>${Object.keys(result.file_breakdown || {}).length}</td></tr>
                                <tr><td><strong>Embeddings:</strong></td><td>${Object.values(result.file_breakdown || {}).reduce((a, b) => a + b, 0)}</td></tr>
                                <tr><td><strong>Vector Dimension:</strong></td><td>${result.embeddings_dimension}</td></tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-info mb-2">Database Size</h6>
                        <table class="table table-sm table-borderless">
                            <tbody>
                                <tr><td><strong>FAISS Index:</strong></td><td>${result.database_size.index_mb} MB</td></tr>
                                <tr><td><strong>Metadata:</strong></td><td>${result.database_size.metadata_kb} KB</td></tr>
                                <tr><td><strong>Total Size:</strong></td><td>${(result.database_size.index_mb + result.database_size.metadata_kb/1024).toFixed(1)} MB</td></tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-info mb-2">Top Domains</h6>
                        <table class="table table-sm table-borderless">
                            <tbody>
            `;

            // Add top domains (first 4)
            const topDomains = Object.entries(result.domain_breakdown).slice(0, 4);
            for (const [domain, count] of topDomains) {
                summaryHTML += `<tr><td><strong>${domain}:</strong></td><td>${count}</td></tr>`;
            }

            summaryHTML += `
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <h6 class="text-info mb-2">Database Breakdown by Sponsor</h6>
                <div class="table-responsive mb-3">
                    <table class="table table-striped">
                        <thead class="table-primary">
                            <tr>
                                <th class="text-dark fw-bold">Sponsor</th>
                                <th class="text-dark fw-bold">Study</th>
                                <th class="text-dark fw-bold">Records</th>
                                <th class="text-dark fw-bold">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            // Add sponsor breakdown with study info
            const sponsorStudyMap = {
                'Abbvie': 'M25-147',
                'Astex': 'ASTX030_01', 
                'Cytokinetics': 'CY-6022',
                'Kura': 'KO-MEN-007',
                'AZ': 'SAAMALIBONC'
            };

            const totalRecords = Object.values(result.file_breakdown || {}).reduce((a, b) => a + b, 0);
            for (const [sponsor, count] of Object.entries(result.sponsor_breakdown)) {
                const percentage = ((count / totalRecords) * 100).toFixed(1);
                const study = sponsorStudyMap[sponsor] || 'Unknown';
                summaryHTML += `
                    <tr>
                        <td><strong>${sponsor}</strong></td>
                        <td>${study}</td>
                        <td>${count}</td>
                        <td>${percentage}%</td>
                    </tr>
                `;
            }

            summaryHTML += `
                        </tbody>
                    </table>
                </div>
            `;

            contentDiv.innerHTML = summaryHTML;
            
            // Show the collapse
            bsCollapse.show();
            
            // Update button text
            summaryBtn.innerHTML = '<i class="fas fa-eye-slash me-2"></i>Hide Database Summary';

        } catch (error) {
            this.showAlert('Failed to load database summary: ' + error.message, 'danger');
        } finally {
            // Reset button
            summaryBtn.disabled = false;
        }
    }

    async checkDatabaseStatus() {
        try {
            const response = await fetch('/database-summary');
            const result = await response.json();
            
            if (result.initialized) {
                this.isDatabaseInitialized = true;
                
                // Update main page status display
                const statusDiv = document.getElementById('dbStatus');
                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Database is ready with ${result.total_records} records from ${result.total_files} files
                    </div>
                `;
                
                // Update main page header status text
                this.updateMainPageHeaderStatus(result.total_records, result.total_files);

                // Update embeddings value in the info card
                const embeddingsValue = document.getElementById('embeddings-value');
                if (embeddingsValue) {
                    embeddingsValue.textContent = result.total_records;
                }
                
                const btn = document.getElementById('initDbBtn');
                btn.innerHTML = '<i class="fas fa-sync me-2"></i>Reinitialize Database';
                btn.classList.remove('btn-outline-primary');
                btn.classList.add('btn-outline-secondary');
                btn.disabled = false;
            }
        } catch (error) {
            console.log('Database not yet initialized');
        }
    }

    updateMainPageHeaderStatus(totalRecords, totalFiles) {
        // Update the main page database status text
        const headerText = document.querySelector('.card-body p.text-muted');
        if (headerText && headerText.textContent.includes('View status of FAISS vector database')) {
            headerText.textContent = `FAISS vector database loaded with OpenAI embeddings from Spec files.`;
        }
    }

    async rebuildDatabase(skipConfirm = false) {
        const rebuildBtn = document.getElementById('rebuildDbBtn');
        const originalText = rebuildBtn.innerHTML;
        
        try {
            // Confirm with user unless skipped (modal flow)
            if (!skipConfirm && !confirm('This will rebuild the vector database with all MDD files in the MDD_DATABASE folder. This may take several minutes. Continue?')) {
                return;
            }
            
            // Update button state
            rebuildBtn.disabled = true;
            rebuildBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Rebuilding Database...';
            
            // Show progress message
            this.showAlert('Rebuilding vector database with current MDD files. This may take several minutes...', 'info');
            
            const response = await fetch('/rebuild-database', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(result.message, 'success');
                
                // Update database status display
                this.updateMainPageHeaderStatus(result.total_records, result.total_files);
                
                // Refresh database summary if it's currently shown
                const summaryDiv = document.getElementById('dbSummary');
                if (summaryDiv.classList.contains('show')) {
                    await this.showDatabaseSummary();
                }
                
                // Mark database as initialized
                this.isDatabaseInitialized = true;
                
            } else {
                this.showAlert('Failed to rebuild database: ' + result.error, 'danger');
            }
            
        } catch (error) {
            this.showAlert('Error rebuilding database: ' + error.message, 'danger');
        } finally {
            // Reset button
            rebuildBtn.disabled = false;
            rebuildBtn.innerHTML = originalText;
        }
    }

    showHelp() {
        const helpBtn = document.getElementById('helpBtn');
        const helpSection = document.getElementById('helpSection');
        
        // Toggle collapse
        const bsCollapse = new bootstrap.Collapse(helpSection, { toggle: false });
        
        // If already shown, just toggle
        if (helpSection.classList.contains('show')) {
            bsCollapse.toggle();
            helpBtn.innerHTML = '<i class="fas fa-question-circle me-2"></i>Help & Documentation';
            return;
        }
        
        // Show the help section
        bsCollapse.show();
        
        // Update button text
        helpBtn.innerHTML = '<i class="fas fa-eye-slash me-2"></i>Hide Help';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MDDAutomationApp();
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Handle file drag and drop
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('targetFile');
    const uploadForm = document.getElementById('uploadForm');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadForm.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadForm.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadForm.classList.add('border-primary');
    }
    
    function unhighlight(e) {
        uploadForm.classList.remove('border-primary');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});

// Add smooth scrolling for anchor links (excluding download buttons)
document.addEventListener('DOMContentLoaded', () => {
    const anchorLinks = document.querySelectorAll('a[href^="#"]:not(#downloadCsv):not(#downloadJson)');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = link.getAttribute('href');
            
            // Skip if href is just "#" or invalid
            if (!targetId || targetId === '#' || targetId.length <= 1) {
                return;
            }
            
            e.preventDefault();
            
            try {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            } catch (selectorError) {
                console.warn('Invalid selector for smooth scrolling:', targetId);
            }
        });
    });
});
