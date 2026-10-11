/**
 * VisionLabel AI — Zero-Shot Vision-Language Image Annotation Studio
 * Frontend Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // Application State
    // =========================================================================
    const state = {
        images: [],
        currentIndex: -1,
        queries: ['dog', 'car', 'person', 'traffic light'],
        similarityThreshold: 0.20,
        iouThreshold: 0.40,
        showBoxes: true,
        showLabels: true,
        toolMode: 'select', // 'select' | 'draw'
        selectedDetectionId: null,
        activeQueryForDraw: null,
        isProcessing: false,

        // Viewport transform
        viewport: {
            zoom: 1.0,
            panX: 0,
            panY: 0,
            isPanning: false,
            startX: 0,
            startY: 0
        },

        // Manual drawing
        drawState: {
            isDrawing: false,
            startX: 0,
            startY: 0,
            tempRect: null
        },

        // Interactive bounding box transformation (Move / Resize)
        transformState: {
            isDragging: false,
            isResizing: false,
            resizeHandle: null,
            targetDet: null,
            startMouseX: 0,
            startMouseY: 0,
            originalBox: null
        }
    };

    // Predefined query color palette
    const QUERY_COLORS = [
        '#00f0ff', // Cyan
        '#10b981', // Emerald
        '#f59e0b', // Amber
        '#8b5cf6', // Purple
        '#f43f5e', // Rose
        '#06b6d4', // Ocean Blue
        '#ec4899', // Pink
        '#84cc16', // Lime
        '#3b82f6', // Royal Blue
        '#d946ef'  // Fuchsia
    ];

    function getQueryColor(index) {
        return QUERY_COLORS[index % QUERY_COLORS.length];
    }

    // =========================================================================
    // DOM Element References
    // =========================================================================
    const elements = {
        // Telemetry & Header
        modelStatusText: document.getElementById('model-status-text'),
        speedMetric: document.getElementById('speed-metric'),
        imageCountMetric: document.getElementById('image-count-metric'),
        badgeTotalImages: document.getElementById('badge-total-images'),
        btnLoadSamples: document.getElementById('btn-load-samples'),
        btnClearAll: document.getElementById('btn-clear-all'),
        btnOpenHelp: document.getElementById('btn-open-help'),
        btnCloseHelp: document.getElementById('btn-close-help'),
        helpModal: document.getElementById('help-modal'),

        // Left Panel (Gallery & Upload)
        dropZone: document.getElementById('drop-zone'),
        fileInput: document.getElementById('file-input'),
        uploadProgressContainer: document.getElementById('upload-progress-container'),
        uploadStatusText: document.getElementById('upload-status-text'),
        uploadPercentText: document.getElementById('upload-percent-text'),
        uploadBarFill: document.getElementById('upload-bar-fill'),
        btnBatchAnnotate: document.getElementById('btn-batch-annotate'),
        btnBatchAnnotateText: document.getElementById('btn-batch-annotate-text'),
        batchProgressContainer: document.getElementById('batch-progress-container'),
        progressStatusText: document.getElementById('progress-status-text'),
        progressPercentText: document.getElementById('progress-percent-text'),
        progressBarFill: document.getElementById('progress-bar-fill'),
        imageGalleryList: document.getElementById('image-gallery-list'),
        galleryEmptyState: document.getElementById('gallery-empty-state'),

        // Center Panel (Canvas Studio)
        canvasImageName: document.getElementById('canvas-image-name'),
        canvasImageDims: document.getElementById('canvas-image-dims'),
        toolSelect: document.getElementById('tool-select'),
        toolDraw: document.getElementById('tool-draw'),
        btnZoomOut: document.getElementById('btn-zoom-out'),
        btnZoomIn: document.getElementById('btn-zoom-in'),
        btnZoomFit: document.getElementById('btn-zoom-fit'),
        zoomLevelText: document.getElementById('zoom-level-text'),
        toggleBoxes: document.getElementById('toggle-boxes'),
        toggleLabels: document.getElementById('toggle-labels'),
        viewportStage: document.getElementById('viewport-stage'),
        canvasTransformer: document.getElementById('canvas-transformer'),
        annotationCanvas: document.getElementById('annotation-canvas'),
        vectorOverlay: document.getElementById('vector-overlay'),
        coordsHud: document.getElementById('coords-hud'),
        
        // Floating Box Inspector & Modifier
        boxInspectorTooltip: document.getElementById('box-inspector-tooltip'),
        tooltipColorDot: document.getElementById('tooltip-color-dot'),
        tooltipLabelInput: document.getElementById('tooltip-label-input'),
        tooltipSim: document.getElementById('tooltip-sim'),
        tooltipConf: document.getElementById('tooltip-conf'),
        boxEditX1: document.getElementById('box-edit-x1'),
        boxEditY1: document.getElementById('box-edit-y1'),
        boxEditW: document.getElementById('box-edit-w'),
        boxEditH: document.getElementById('box-edit-h'),
        btnSaveBoxEdit: document.getElementById('btn-save-box-edit'),
        tooltipDeleteBtn: document.getElementById('tooltip-delete-btn'),
        
        btnPrevImage: document.getElementById('btn-prev-image'),
        btnNextImage: document.getElementById('btn-next-image'),
        imagePaginationText: document.getElementById('image-pagination-text'),

        // Right Panel (Queries & Inspector)
        queryInput: document.getElementById('query-input'),
        btnAddQuery: document.getElementById('btn-add-query'),
        queryChipsList: document.getElementById('query-chips-list'),
        btnDetectCurrent: document.getElementById('btn-detect-current'),
        similaritySlider: document.getElementById('similarity-slider'),
        similarityValue: document.getElementById('similarity-value'),
        iouSlider: document.getElementById('iou-slider'),
        iouValue: document.getElementById('iou-value'),
        detectionsCountBadge: document.getElementById('detections-count-badge'),
        detectionsListContainer: document.getElementById('detections-list-container'),
        detectionsEmptyState: document.getElementById('detections-empty-state'),

        // Export Actions
        btnExportCoco: document.getElementById('btn-export-coco'),
        btnExportYolo: document.getElementById('btn-export-yolo'),
        btnExportJson: document.getElementById('btn-export-json'),
        btnExportImages: document.getElementById('btn-export-images'),
        btnSaveCurrent: document.getElementById('btn-save-current')
    };

    // Off-screen image object for current active image
    let currentImageElement = new Image();

    // =========================================================================
    // Initialization
    // =========================================================================
    function init() {
        renderQueryChips();
        setupEventListeners();
        checkBackendStatus();
    }

    async function checkBackendStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.status === 'ready') {
                elements.modelStatusText.textContent = 'Engine Ready (Fast VL)';
            }
        } catch (e) {
            elements.modelStatusText.textContent = 'Connecting to Server...';
        }
    }

    // =========================================================================
    // Query Management
    // =========================================================================
    function renderQueryChips() {
        elements.queryChipsList.innerHTML = '';
        state.queries.forEach((q, idx) => {
            const color = getQueryColor(idx);
            const chip = document.createElement('div');
            chip.className = 'query-chip';
            chip.innerHTML = `
                <span class="query-chip-color" style="background-color: ${color}"></span>
                <span>${escapeHtml(q)}</span>
                <button class="query-chip-remove" title="Remove query">&times;</button>
            `;
            chip.querySelector('.query-chip-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                removeQuery(q);
            });
            elements.queryChipsList.appendChild(chip);
        });

        if (state.queries.length > 0 && !state.activeQueryForDraw) {
            state.activeQueryForDraw = state.queries[0];
        }
    }

    function addQueries(queryStr) {
        if (!queryStr || !queryStr.trim()) return;
        const parts = queryStr.split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
        let added = false;
        parts.forEach(p => {
            if (!state.queries.includes(p)) {
                state.queries.push(p);
                added = true;
            }
        });
        if (added) {
            renderQueryChips();
        }
        elements.queryInput.value = '';
    }

    function removeQuery(query) {
        state.queries = state.queries.filter(q => q !== query);
        renderQueryChips();
    }

    // Preset Buttons
    document.querySelectorAll('.preset-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const presetStr = btn.getAttribute('data-preset');
            state.queries = []; // Replace with preset
            addQueries(presetStr);
        });
    });

    // =========================================================================
    // Batch Upload & Gallery (Supports up to 1000 images)
    // =========================================================================
    async function uploadFiles(filesList) {
        if (!filesList || filesList.length === 0) return;
        
        let files = Array.from(filesList);
        if (files.length > 1000) {
            alert(`You selected ${files.length} images. Limit is 1,000 at once; uploading the first 1,000 images.`);
            files = files.slice(0, 1000);
        }

        const totalFiles = files.length;
        const chunkSize = 40; // upload in chunks of 40 images for smooth progress and network stability
        const totalChunks = Math.ceil(totalFiles / chunkSize);

        if (elements.uploadProgressContainer) {
            elements.uploadProgressContainer.classList.remove('hidden');
            elements.uploadStatusText.textContent = `Preparing upload of ${totalFiles} images...`;
            elements.uploadPercentText.textContent = `0%`;
            elements.uploadBarFill.style.width = `0%`;
        }
        elements.modelStatusText.textContent = `Uploading ${totalFiles} images...`;

        let uploadedSoFar = 0;

        for (let c = 0; c < totalChunks; c++) {
            const chunk = files.slice(c * chunkSize, (c + 1) * chunkSize);
            const formData = new FormData();
            for (let i = 0; i < chunk.length; i++) {
                formData.append('files', chunk[i]);
            }

            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.success && data.images) {
                    appendImages(data.images);
                }
            } catch (err) {
                console.error(`Upload error on batch ${c + 1}:`, err);
            }

            uploadedSoFar += chunk.length;
            const pct = Math.round((uploadedSoFar / totalFiles) * 100);
            if (elements.uploadProgressContainer) {
                elements.uploadStatusText.textContent = `Uploading: ${uploadedSoFar} / ${totalFiles} images (Batch ${c + 1}/${totalChunks})...`;
                elements.uploadPercentText.textContent = `${pct}%`;
                elements.uploadBarFill.style.width = `${pct}%`;
            }
        }

        if (elements.uploadProgressContainer) {
            elements.uploadStatusText.textContent = `Upload complete! ${totalFiles} images ready.`;
            elements.uploadPercentText.textContent = `100%`;
            elements.uploadBarFill.style.width = `100%`;
            setTimeout(() => {
                elements.uploadProgressContainer.classList.add('hidden');
            }, 2500);
        }
        elements.modelStatusText.textContent = 'Engine Ready (Fast VL)';
    }

    async function loadSamplePack() {
        elements.modelStatusText.textContent = 'Loading sample pack...';
        try {
            const res = await fetch('/api/load_all_samples', { method: 'POST' });
            const data = await res.json();
            if (data.success && data.images) {
                state.images = [];
                appendImages(data.images);
                // Also set rich initial queries matching the samples
                state.queries = ['dog', 'car', 'traffic light', 'headphones', 'sink', 'cat'];
                renderQueryChips();
            }
        } catch (err) {
            console.error('Failed to load samples:', err);
        } finally {
            elements.modelStatusText.textContent = 'Engine Ready (Fast VL)';
        }
    }

    function appendImages(newImages) {
        newImages.forEach(img => {
            state.images.push(img);
        });

        updateHeaderStats();
        renderGalleryList();

        if (state.currentIndex === -1 && state.images.length > 0) {
            setActiveImage(0);
        }

        elements.btnBatchAnnotate.disabled = state.images.length === 0;
        elements.btnDetectCurrent.disabled = state.images.length === 0;
    }

    function renderGalleryList() {
        elements.imageGalleryList.innerHTML = '';
        if (state.images.length === 0) {
            elements.imageGalleryList.appendChild(elements.galleryEmptyState);
            return;
        }

        state.images.forEach((img, idx) => {
            const card = document.createElement('div');
            card.className = `image-card ${idx === state.currentIndex ? 'active' : ''}`;
            const isAnnotated = img.detections && img.detections.length > 0;
            const detCount = img.detections ? img.detections.length : 0;

            card.innerHTML = `
                <div class="image-card-thumb-wrap">
                    <img class="image-card-thumb" src="${img.url}" alt="${escapeHtml(img.filename)}" loading="lazy">
                </div>
                <div class="image-card-info">
                    <span class="image-card-name" title="${escapeHtml(img.filename)}">${escapeHtml(img.filename)}</span>
                    <div class="image-card-meta">
                        <span>${img.width}×${img.height}</span>
                        <span class="card-status-pill ${isAnnotated ? 'annotated' : 'unprocessed'}">
                            ${isAnnotated ? `${detCount} objects` : 'Unprocessed'}
                        </span>
                    </div>
                </div>
                <button class="btn-card-delete" title="Remove image">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            `;

            card.addEventListener('click', () => {
                setActiveImage(idx);
            });

            card.querySelector('.btn-card-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteImage(idx);
            });

            elements.imageGalleryList.appendChild(card);
        });
    }

    function updateHeaderStats() {
        const count = state.images.length;
        elements.badgeTotalImages.textContent = count;
        elements.imageCountMetric.textContent = `${count} Image${count === 1 ? '' : 's'} Loaded`;
        elements.btnBatchAnnotateText.textContent = `Annotate All ${count} Images`;
    }

    async function deleteImage(index) {
        const targetImg = state.images[index];
        if (!targetImg) return;

        try {
            await fetch(`/api/image/${targetImg.id}`, { method: 'DELETE' });
        } catch (e) {
            console.error('Delete request failed:', e);
        }

        state.images.splice(index, 1);
        if (state.currentIndex >= state.images.length) {
            state.currentIndex = state.images.length - 1;
        }

        updateHeaderStats();
        renderGalleryList();

        if (state.currentIndex >= 0) {
            setActiveImage(state.currentIndex);
        } else {
            clearCanvas();
        }

        elements.btnBatchAnnotate.disabled = state.images.length === 0;
        elements.btnDetectCurrent.disabled = state.images.length === 0;
    }

    function clearAll() {
        if (!confirm('Clear all loaded images from session?')) return;
        fetch('/api/clear', { method: 'POST' });
        state.images = [];
        state.currentIndex = -1;
        updateHeaderStats();
        renderGalleryList();
        clearCanvas();
        elements.btnBatchAnnotate.disabled = true;
        elements.btnDetectCurrent.disabled = true;
    }

    // =========================================================================
    // Active Image & Canvas Rendering
    // =========================================================================
    function setActiveImage(index) {
        if (index < 0 || index >= state.images.length) return;
        state.currentIndex = index;
        const current = state.images[index];

        elements.canvasImageName.textContent = current.filename;
        elements.canvasImageDims.textContent = `${current.width} × ${current.height}`;
        elements.imagePaginationText.textContent = `Image ${index + 1} of ${state.images.length}`;

        // Highlight card in gallery
        document.querySelectorAll('.image-card').forEach((card, i) => {
            card.classList.toggle('active', i === index);
        });

        // Load image into canvas
        currentImageElement.onload = () => {
            elements.annotationCanvas.width = current.width;
            elements.annotationCanvas.height = current.height;
            const ctx = elements.annotationCanvas.getContext('2d');
            ctx.drawImage(currentImageElement, 0, 0);

            // Set vector overlay viewBox
            elements.vectorOverlay.setAttribute('viewBox', `0 0 ${current.width} ${current.height}`);
            elements.vectorOverlay.setAttribute('width', current.width);
            elements.vectorOverlay.setAttribute('height', current.height);

            fitToViewport();
            renderVectorAnnotations();
            renderDetectionsInspector();
        };
        currentImageElement.src = current.url;

        // Hide floating tooltip
        hideTooltip();
    }

    function clearCanvas() {
        elements.canvasImageName.textContent = 'No image selected';
        elements.canvasImageDims.textContent = '0 × 0';
        elements.imagePaginationText.textContent = 'Image 0 of 0';
        const ctx = elements.annotationCanvas.getContext('2d');
        ctx.clearRect(0, 0, elements.annotationCanvas.width, elements.annotationCanvas.height);
        elements.vectorOverlay.innerHTML = '';
        renderDetectionsInspector();
        hideTooltip();
    }

    async function syncDetectionsToBackend() {
        if (state.currentIndex === -1) return;
        const current = state.images[state.currentIndex];
        if (!current) return;
        try {
            await fetch(`/api/detections/${current.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detections: current.detections })
            });
        } catch (err) {
            console.error('Failed to sync detections:', err);
        }
    }

    function renderVectorAnnotations() {
        elements.vectorOverlay.innerHTML = '';
        if (state.currentIndex === -1 || !state.showBoxes) return;

        const current = state.images[state.currentIndex];
        if (!current || !current.detections) return;

        const filtered = current.detections.filter(d => d.similarity >= state.similarityThreshold);
        const zoom = state.viewport.zoom || 1.0;

        // Size calculation for handles and borders in SVG coordinates:
        // Scaled inversely with zoom so physical screen size remains constant (~10px diameter, ~24px hit area)
        const handleRadius = Math.max(3, Math.min(16, 5 / zoom));
        const hitRadius = Math.max(8, Math.min(32, 12 / zoom));
        const handleStrokeWidth = Math.max(1.5, Math.min(5, 2 / zoom));
        const edgeHitWidth = Math.max(8, Math.min(32, 12 / zoom));

        filtered.forEach(det => {
            const [x1, y1, x2, y2] = det.box;
            const bw = Math.max(1, x2 - x1);
            const bh = Math.max(1, y2 - y1);
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;
            const color = det.color || '#00f0ff';
            const isSelected = det.id === state.selectedDetectionId;

            // Group container
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('class', `bbox-group ${isSelected ? 'selected' : ''}`);
            g.setAttribute('data-id', det.id);

            // Bounding box rect
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x1);
            rect.setAttribute('y', y1);
            rect.setAttribute('width', bw);
            rect.setAttribute('height', bh);
            rect.setAttribute('stroke', color);
            rect.setAttribute('class', `bbox-rect ${isSelected ? 'selected' : ''}`);

            // Mousedown for moving/translating
            rect.addEventListener('mousedown', (e) => {
                if (e.button !== 0) return; // Left click only
                e.stopPropagation();
                if (state.selectedDetectionId !== det.id) {
                    selectDetection(det.id);
                }
                startMoving(det, e);
            });

            rect.addEventListener('click', (e) => {
                if (e.button !== 0) return;
                e.stopPropagation();
                if (state.selectedDetectionId !== det.id) {
                    selectDetection(det.id);
                }
            });

            rect.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                promptRenameDetection(det);
            });

            g.appendChild(rect);

            // 4 Side Edge Hit-Zones (Top, Bottom, Left, Right)
            // Allows modifying box sides directly by clicking and dragging anywhere on that side
            const edgeDefs = [
                { type: 'n', x1: x1, y1: y1, x2: x2, y2: y1, cursor: 'ns-resize' },
                { type: 's', x1: x1, y1: y2, x2: x2, y2: y2, cursor: 'ns-resize' },
                { type: 'w', x1: x1, y1: y1, x2: x1, y2: y2, cursor: 'ew-resize' },
                { type: 'e', x1: x2, y1: y1, x2: x2, y2: y2, cursor: 'ew-resize' }
            ];

            edgeDefs.forEach(edge => {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', edge.x1);
                line.setAttribute('y1', edge.y1);
                line.setAttribute('x2', edge.x2);
                line.setAttribute('y2', edge.y2);
                line.setAttribute('stroke', 'transparent');
                line.setAttribute('stroke-width', edgeHitWidth);
                line.setAttribute('class', `bbox-edge-hit edge-${edge.type}`);
                line.style.cursor = edge.cursor;

                line.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return; // Left click only
                    e.stopPropagation();
                    if (state.selectedDetectionId !== det.id) {
                        selectDetection(det.id);
                    }
                    startResizing(det, edge.type, e);
                });

                g.appendChild(line);
            });

            // 8 Interactive Resize Handles: 4 middle points of the sides + 4 corners
            // When selected, prominent handles allow intuitive left-click modification just like all other annotation tools
            if (isSelected) {
                const handles = [
                    // 4 Middle points of the sides
                    { type: 'n',  x: midX, y: y1,   cursor: 'ns-resize',   isMiddle: true, title: 'Drag to resize top edge' },
                    { type: 's',  x: midX, y: y2,   cursor: 'ns-resize',   isMiddle: true, title: 'Drag to resize bottom edge' },
                    { type: 'w',  x: x1,   y: midY, cursor: 'ew-resize',   isMiddle: true, title: 'Drag to resize left edge' },
                    { type: 'e',  x: x2,   y: midY, cursor: 'ew-resize',   isMiddle: true, title: 'Drag to resize right edge' },
                    // 4 Corner points
                    { type: 'nw', x: x1,   y: y1,   cursor: 'nwse-resize', isMiddle: false, title: 'Drag to resize corner' },
                    { type: 'ne', x: x2,   y: y1,   cursor: 'nesw-resize', isMiddle: false, title: 'Drag to resize corner' },
                    { type: 'se', x: x2,   y: y2,   cursor: 'nwse-resize', isMiddle: false, title: 'Drag to resize corner' },
                    { type: 'sw', x: x1,   y: y2,   cursor: 'nesw-resize', isMiddle: false, title: 'Drag to resize corner' }
                ];

                handles.forEach(h => {
                    // Larger invisible hit circle for effortless clicking
                    const hitCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    hitCircle.setAttribute('cx', h.x);
                    hitCircle.setAttribute('cy', h.y);
                    hitCircle.setAttribute('r', hitRadius);
                    hitCircle.setAttribute('class', `bbox-handle-hit handle-hit-${h.type}`);
                    hitCircle.style.cursor = h.cursor;

                    // Visible handle dot (middle handles are slightly larger and styled)
                    const handleCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    handleCircle.setAttribute('cx', h.x);
                    handleCircle.setAttribute('cy', h.y);
                    handleCircle.setAttribute('r', h.isMiddle ? handleRadius * 1.15 : handleRadius);
                    handleCircle.setAttribute('fill', '#ffffff');
                    handleCircle.setAttribute('stroke', color);
                    handleCircle.setAttribute('stroke-width', handleStrokeWidth);
                    handleCircle.setAttribute('class', `bbox-handle handle-${h.type} ${h.isMiddle ? 'handle-middle' : 'handle-corner'}`);
                    handleCircle.style.cursor = h.cursor;

                    const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                    titleEl.textContent = h.title;
                    hitCircle.appendChild(titleEl);

                    // Left-click on middle point or corner to modify bounding box
                    hitCircle.addEventListener('mousedown', (e) => {
                        if (e.button !== 0) return; // Left click only
                        e.stopPropagation();
                        startResizing(det, h.type, e);
                    });

                    handleCircle.addEventListener('mousedown', (e) => {
                        if (e.button !== 0) return;
                        e.stopPropagation();
                        startResizing(det, h.type, e);
                    });

                    g.appendChild(hitCircle);
                    g.appendChild(handleCircle);
                });
            }

            // Label badge
            if (state.showLabels) {
                const labelText = `${det.query} ${Math.round(det.similarity * 100)}%`;
                const textLen = labelText.length * 7.5 + 12;
                const badgeH = 18;
                const badgeY = Math.max(0, y1 - badgeH);

                const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                bgRect.setAttribute('x', x1);
                bgRect.setAttribute('y', badgeY);
                bgRect.setAttribute('width', textLen);
                bgRect.setAttribute('height', badgeH);
                bgRect.setAttribute('fill', color);
                bgRect.setAttribute('rx', '3');
                bgRect.style.cursor = 'pointer';

                bgRect.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    e.stopPropagation();
                    if (state.selectedDetectionId !== det.id) {
                        selectDetection(det.id);
                    }
                    startMoving(det, e);
                });

                bgRect.addEventListener('dblclick', (e) => {
                    e.stopPropagation();
                    promptRenameDetection(det);
                });

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x1 + 6);
                text.setAttribute('y', badgeY + 13);
                text.setAttribute('class', 'bbox-label-badge');
                text.textContent = labelText;
                text.style.cursor = 'pointer';

                text.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    e.stopPropagation();
                    if (state.selectedDetectionId !== det.id) {
                        selectDetection(det.id);
                    }
                    startMoving(det, e);
                });

                text.addEventListener('dblclick', (e) => {
                    e.stopPropagation();
                    promptRenameDetection(det);
                });

                g.appendChild(bgRect);
                g.appendChild(text);
            }

            elements.vectorOverlay.appendChild(g);
        });

        elements.detectionsCountBadge.textContent = filtered.length;
    }

    function startMoving(det, e) {
        if (e.button !== 0) return; // Left click only
        const stageRect = elements.viewportStage.getBoundingClientRect();
        const curImgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
        const curImgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;

        state.transformState.isDragging = true;
        state.transformState.isResizing = false;
        state.transformState.targetDet = det;
        state.transformState.startMouseX = curImgX;
        state.transformState.startMouseY = curImgY;
        state.transformState.originalBox = [...det.box];
    }

    function startResizing(det, handleType, e) {
        if (e.button !== 0) return; // Left click only
        const stageRect = elements.viewportStage.getBoundingClientRect();
        const curImgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
        const curImgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;

        state.transformState.isResizing = true;
        state.transformState.isDragging = false;
        state.transformState.targetDet = det;
        state.transformState.resizeHandle = handleType;
        state.transformState.startMouseX = curImgX;
        state.transformState.startMouseY = curImgY;
        state.transformState.originalBox = [...det.box];
    }

    function selectDetection(detId) {
        state.selectedDetectionId = detId;
        renderVectorAnnotations();

        // Highlight inspector item in the right sidebar
        document.querySelectorAll('.detection-item-card').forEach(card => {
            const isMatch = card.getAttribute('data-id') === detId;
            card.classList.toggle('highlighted', isMatch);
            if (isMatch) {
                card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
        });

        // Keep popup tooltip window hidden (replaced by direct on-canvas handles)
        if (elements.boxInspectorTooltip) {
            elements.boxInspectorTooltip.classList.add('hidden');
        }
    }

    function showTooltip(det) {
        // Disabled: user requested replacing popup modal with direct handle manipulation
        if (elements.boxInspectorTooltip) {
            elements.boxInspectorTooltip.classList.add('hidden');
        }
    }

    function updateTooltipFields(det) {
        // Safely update input fields if elements exist in DOM
        if (!det) return;
        const [x1, y1, x2, y2] = det.box;
        if (elements.boxEditX1) elements.boxEditX1.value = Math.round(x1);
        if (elements.boxEditY1) elements.boxEditY1.value = Math.round(y1);
        if (elements.boxEditW) elements.boxEditW.value = Math.round(x2 - x1);
        if (elements.boxEditH) elements.boxEditH.value = Math.round(y2 - y1);
    }

    function promptRenameDetection(det) {
        const cur = det.query || '';
        const updated = prompt('Edit object label:', cur);
        if (updated !== null && updated.trim() !== '') {
            const clean = updated.trim().toLowerCase();
            det.query = clean;
            let qIdx = state.queries.indexOf(clean);
            if (qIdx === -1) {
                state.queries.push(clean);
                renderQueryChips();
                qIdx = state.queries.length - 1;
            }
            det.color = getQueryColor(qIdx);
            renderVectorAnnotations();
            renderDetectionsInspector();
            syncDetectionsToBackend();
        }
    }

    function applyManualCoordInputs() {
        if (!state.selectedDetectionId || state.currentIndex === -1) return;
        const current = state.images[state.currentIndex];
        const det = current.detections.find(d => d.id === state.selectedDetectionId);
        if (!det) return;

        const maxW = current.width || 5000;
        const maxH = current.height || 5000;

        const x1 = Math.max(0, Math.min(parseInt(elements.boxEditX1.value) || 0, maxW - 10));
        const y1 = Math.max(0, Math.min(parseInt(elements.boxEditY1.value) || 0, maxH - 10));
        const w = Math.max(5, Math.min(parseInt(elements.boxEditW.value) || 10, maxW - x1));
        const h = Math.max(5, Math.min(parseInt(elements.boxEditH.value) || 10, maxH - y1));

        det.box = [Math.round(x1), Math.round(y1), Math.round(x1 + w), Math.round(y1 + h)];
        renderVectorAnnotations();
        renderDetectionsInspector();
        syncDetectionsToBackend();
    }

    function hideTooltip() {
        if (elements.boxInspectorTooltip) {
            elements.boxInspectorTooltip.classList.add('hidden');
        }
        state.selectedDetectionId = null;
    }

    function deleteSelectedDetection() {
        if (!state.selectedDetectionId || state.currentIndex === -1) return;
        const current = state.images[state.currentIndex];
        current.detections = current.detections.filter(d => d.id !== state.selectedDetectionId);
        hideTooltip();
        renderVectorAnnotations();
        renderDetectionsInspector();
        renderGalleryList();
        syncDetectionsToBackend();
    }

    function renderDetectionsInspector() {
        elements.detectionsListContainer.innerHTML = '';
        if (state.currentIndex === -1) {
            elements.detectionsListContainer.appendChild(elements.detectionsEmptyState);
            elements.detectionsCountBadge.textContent = '0';
            return;
        }

        const current = state.images[state.currentIndex];
        const filtered = (current.detections || []).filter(d => d.similarity >= state.similarityThreshold);
        elements.detectionsCountBadge.textContent = filtered.length;

        if (filtered.length === 0) {
            elements.detectionsListContainer.appendChild(elements.detectionsEmptyState);
            return;
        }

        filtered.forEach(det => {
            const card = document.createElement('div');
            card.className = `detection-item-card ${det.id === state.selectedDetectionId ? 'highlighted' : ''}`;
            card.setAttribute('data-id', det.id);
            card.innerHTML = `
                <div class="detection-item-left">
                    <span class="detection-dot" style="background-color: ${det.color || '#00f0ff'}"></span>
                    <span class="detection-label" title="Double-click to rename label">${escapeHtml(det.query)}</span>
                </div>
                <div class="detection-item-left">
                    <span class="detection-score">${Math.round(det.similarity * 100)}%</span>
                    <button class="btn-del-detection" title="Delete detection">&times;</button>
                </div>
            `;

            card.addEventListener('click', () => {
                selectDetection(det.id);
            });

            card.querySelector('.detection-label').addEventListener('dblclick', (e) => {
                e.stopPropagation();
                promptRenameDetection(det);
            });

            card.querySelector('.btn-del-detection').addEventListener('click', (e) => {
                e.stopPropagation();
                current.detections = current.detections.filter(d => d.id !== det.id);
                if (state.selectedDetectionId === det.id) {
                    state.selectedDetectionId = null;
                }
                renderVectorAnnotations();
                renderDetectionsInspector();
                renderGalleryList();
                syncDetectionsToBackend();
            });

            elements.detectionsListContainer.appendChild(card);
        });
    }

    // =========================================================================
    // Viewport Panning & Zooming
    // =========================================================================
    function updateTransform() {
        const { zoom, panX, panY } = state.viewport;
        elements.canvasTransformer.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
        elements.zoomLevelText.textContent = `${Math.round(zoom * 100)}%`;
        if (state.selectedDetectionId) {
            renderVectorAnnotations();
        }
    }

    function fitToViewport() {
        if (state.currentIndex === -1) return;
        const current = state.images[state.currentIndex];
        const stageW = elements.viewportStage.clientWidth;
        const stageH = elements.viewportStage.clientHeight;

        if (stageW === 0 || stageH === 0) return;

        const padding = 40;
        const scaleX = (stageW - padding) / current.width;
        const scaleY = (stageH - padding) / current.height;
        const zoom = Math.min(scaleX, scaleY, 1.0);

        const panX = (stageW - current.width * zoom) / 2;
        const panY = (stageH - current.height * zoom) / 2;

        state.viewport.zoom = zoom;
        state.viewport.panX = panX;
        state.viewport.panY = panY;
        updateTransform();
    }

    function zoomBy(factor) {
        const stageW = elements.viewportStage.clientWidth;
        const stageH = elements.viewportStage.clientHeight;
        const centerX = stageW / 2;
        const centerY = stageH / 2;

        const newZoom = Math.max(0.1, Math.min(state.viewport.zoom * factor, 8.0));
        state.viewport.panX = centerX - (centerX - state.viewport.panX) * (newZoom / state.viewport.zoom);
        state.viewport.panY = centerY - (centerY - state.viewport.panY) * (newZoom / state.viewport.zoom);
        state.viewport.zoom = newZoom;
        updateTransform();
    }

    // =========================================================================
    // Autonomous Detection Engine Execution
    // =========================================================================
    async function runDetectionOnCurrent() {
        if (state.currentIndex === -1 || state.isProcessing) return;
        const current = state.images[state.currentIndex];
        if (!current) return;

        if (state.queries.length === 0) {
            alert('Please add at least one query prompt (e.g. "dog", "car") first.');
            return;
        }

        state.isProcessing = true;
        elements.btnDetectCurrent.disabled = true;
        elements.modelStatusText.textContent = `Detecting on ${current.filename}...`;

        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: current.id,
                    queries: state.queries,
                    similarity_threshold: state.similarityThreshold,
                    iou_threshold: state.iouThreshold
                })
            });

            const data = await res.json();
            if (data.success) {
                current.detections = data.detections;
                elements.speedMetric.textContent = `${data.retrieval_latency_ms} ms Retrieval`;
                renderVectorAnnotations();
                renderDetectionsInspector();
                renderGalleryList();
            } else {
                alert('Detection error: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Detection request failed:', err);
            alert('Detection request failed: ' + err.message);
        } finally {
            state.isProcessing = false;
            elements.btnDetectCurrent.disabled = false;
            elements.modelStatusText.textContent = 'Engine Ready (Fast VL)';
        }
    }

    async function runBatchAnnotation() {
        if (state.images.length === 0 || state.isProcessing) return;
        if (state.queries.length === 0) {
            alert('Please add at least one target query prompt first.');
            return;
        }

        state.isProcessing = true;
        elements.btnBatchAnnotate.disabled = true;
        elements.btnDetectCurrent.disabled = true;
        elements.batchProgressContainer.classList.remove('hidden');

        const total = state.images.length;
        let totalObjects = 0;
        let totalTimeMs = 0;

        for (let i = 0; i < total; i++) {
            const img = state.images[i];
            const pct = Math.round(((i) / total) * 100);
            elements.progressStatusText.textContent = `Processing image ${i + 1} of ${total} (${img.filename})...`;
            elements.progressPercentText.textContent = `${pct}%`;
            elements.progressBarFill.style.width = `${pct}%`;

            try {
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_id: img.id,
                        queries: state.queries,
                        similarity_threshold: state.similarityThreshold,
                        iou_threshold: state.iouThreshold
                    })
                });
                const data = await res.json();
                if (data.success) {
                    img.detections = data.detections;
                    totalObjects += data.detections.length;
                    totalTimeMs += data.total_latency_ms;
                }
            } catch (err) {
                console.error(`Error on image ${img.id}:`, err);
            }

            renderGalleryList();
            if (i === state.currentIndex) {
                renderVectorAnnotations();
                renderDetectionsInspector();
            }
        }

        // Completion
        elements.progressBarFill.style.width = '100%';
        elements.progressPercentText.textContent = '100%';
        elements.progressStatusText.textContent = `Completed! ${totalObjects} objects annotated across ${total} images.`;
        elements.speedMetric.textContent = `${Math.round(totalTimeMs / total)} ms Avg Latency`;

        setTimeout(() => {
            elements.batchProgressContainer.classList.add('hidden');
        }, 3000);

        state.isProcessing = false;
        elements.btnBatchAnnotate.disabled = false;
        elements.btnDetectCurrent.disabled = false;
    }

    // =========================================================================
    // Export Functions (COCO, YOLO, JSON, ZIP Images)
    // =========================================================================
    function triggerDownload(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    async function exportCoco() {
        try {
            const res = await fetch('/api/export/coco');
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            triggerDownload(URL.createObjectURL(blob), 'coco_annotations.json');
        } catch (e) {
            alert('Export failed: ' + e.message);
        }
    }

    function exportYolo() {
        triggerDownload('/api/export/yolo', 'yolo_annotations.zip');
    }

    async function exportJson() {
        try {
            const res = await fetch('/api/export/json');
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            triggerDownload(URL.createObjectURL(blob), 'visionlabel_annotations.json');
        } catch (e) {
            alert('Export failed: ' + e.message);
        }
    }

    function exportAnnotatedImages() {
        triggerDownload('/api/export/annotated_images', 'annotated_images.zip');
    }

    function downloadCurrentImage() {
        if (state.currentIndex === -1) return;
        const current = state.images[state.currentIndex];
        
        // Render onto an off-screen canvas with drawn bounding boxes
        const offCanvas = document.createElement('canvas');
        offCanvas.width = current.width;
        offCanvas.height = current.height;
        const ctx = offCanvas.getContext('2d');
        ctx.drawImage(currentImageElement, 0, 0);

        // Draw boxes
        const filtered = (current.detections || []).filter(d => d.similarity >= state.similarityThreshold);
        ctx.lineWidth = 3;
        ctx.font = 'bold 14px monospace';

        filtered.forEach(det => {
            const [x1, y1, x2, y2] = det.box;
            const color = det.color || '#00f0ff';
            ctx.strokeStyle = color;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Badge
            const label = `${det.query} ${Math.round(det.similarity * 100)}%`;
            const badgeW = ctx.measureText(label).width + 12;
            const badgeH = 20;
            const badgeY = Math.max(0, y1 - badgeH);

            ctx.fillStyle = color;
            ctx.fillRect(x1, badgeY, badgeW, badgeH);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x1 + 6, badgeY + 15);
        });

        offCanvas.toBlob(blob => {
            triggerDownload(URL.createObjectURL(blob), `annotated_${current.filename}`);
        }, 'image/jpeg', 0.95);
    }

    // =========================================================================
    // Event Listeners Setup
    // =========================================================================
    function setupEventListeners() {
        // Query adding
        elements.btnAddQuery.addEventListener('click', () => {
            addQueries(elements.queryInput.value);
        });
        elements.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addQueries(elements.queryInput.value);
            }
        });

        // Threshold Sliders (Instant reactive update)
        elements.similaritySlider.addEventListener('input', (e) => {
            state.similarityThreshold = parseFloat(e.target.value);
            elements.similarityValue.textContent = state.similarityThreshold.toFixed(2);
            renderVectorAnnotations();
            renderDetectionsInspector();
        });

        elements.iouSlider.addEventListener('input', (e) => {
            state.iouThreshold = parseFloat(e.target.value);
            elements.iouValue.textContent = state.iouThreshold.toFixed(2);
        });

        // Toggles
        elements.toggleBoxes.addEventListener('change', (e) => {
            state.showBoxes = e.target.checked;
            renderVectorAnnotations();
        });
        elements.toggleLabels.addEventListener('change', (e) => {
            state.showLabels = e.target.checked;
            renderVectorAnnotations();
        });

        // Drag & Drop
        elements.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.dropZone.classList.add('drag-active');
        });
        elements.dropZone.addEventListener('dragleave', () => {
            elements.dropZone.classList.remove('drag-active');
        });
        elements.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.dropZone.classList.remove('drag-active');
            if (e.dataTransfer.files) {
                uploadFiles(e.dataTransfer.files);
            }
        });
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files) {
                uploadFiles(e.target.files);
            }
        });

        // Sample loading & clearing
        elements.btnLoadSamples.addEventListener('click', loadSamplePack);
        elements.btnClearAll.addEventListener('click', clearAll);

        // Actions
        elements.btnDetectCurrent.addEventListener('click', runDetectionOnCurrent);
        elements.btnBatchAnnotate.addEventListener('click', runBatchAnnotation);

        // Export
        elements.btnExportCoco.addEventListener('click', exportCoco);
        elements.btnExportYolo.addEventListener('click', exportYolo);
        elements.btnExportJson.addEventListener('click', exportJson);
        elements.btnExportImages.addEventListener('click', exportAnnotatedImages);
        elements.btnSaveCurrent.addEventListener('click', downloadCurrentImage);

        // Zoom & Fit
        elements.btnZoomIn.addEventListener('click', () => zoomBy(1.2));
        elements.btnZoomOut.addEventListener('click', () => zoomBy(0.833));
        elements.btnZoomFit.addEventListener('click', fitToViewport);

        // Tool Mode selection
        elements.toolSelect.addEventListener('click', () => {
            state.toolMode = 'select';
            elements.toolSelect.classList.add('active');
            elements.toolDraw.classList.remove('active');
            elements.viewportStage.classList.remove('drawing');
        });

        elements.toolDraw.addEventListener('click', () => {
            state.toolMode = 'draw';
            elements.toolDraw.classList.add('active');
            elements.toolSelect.classList.remove('active');
            elements.viewportStage.classList.add('drawing');
        });

        // Mouse Pan & Zoom in Canvas Viewport
        elements.viewportStage.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.87;
            const stageRect = elements.viewportStage.getBoundingClientRect();
            const mouseX = e.clientX - stageRect.left;
            const mouseY = e.clientY - stageRect.top;

            const newZoom = Math.max(0.1, Math.min(state.viewport.zoom * factor, 8.0));
            state.viewport.panX = mouseX - (mouseX - state.viewport.panX) * (newZoom / state.viewport.zoom);
            state.viewport.panY = mouseY - (mouseY - state.viewport.panY) * (newZoom / state.viewport.zoom);
            state.viewport.zoom = newZoom;
            updateTransform();
        });

        elements.viewportStage.addEventListener('mousedown', (e) => {
            if (e.target.closest('.bbox-rect') || e.target.closest('.bbox-edge-hit') || e.target.closest('.bbox-handle') || e.target.closest('.bbox-handle-hit')) {
                return;
            }

            if (state.toolMode === 'draw') {
                if (state.selectedDetectionId) {
                    state.selectedDetectionId = null;
                    renderVectorAnnotations();
                    renderDetectionsInspector();
                }

                // Start drawing box
                const stageRect = elements.viewportStage.getBoundingClientRect();
                const imgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
                const imgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;
                state.drawState.isDrawing = true;
                state.drawState.startX = imgX;
                state.drawState.startY = imgY;

                const tempRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                tempRect.setAttribute('x', imgX);
                tempRect.setAttribute('y', imgY);
                tempRect.setAttribute('width', 0);
                tempRect.setAttribute('height', 0);
                tempRect.setAttribute('stroke', getQueryColor(0));
                tempRect.setAttribute('class', 'bbox-rect selected');
                elements.vectorOverlay.appendChild(tempRect);
                state.drawState.tempRect = tempRect;
                return;
            }

            // Clicked empty stage: deselect
            if (state.selectedDetectionId) {
                state.selectedDetectionId = null;
                renderVectorAnnotations();
                renderDetectionsInspector();
            }

            // Otherwise pan
            state.viewport.isPanning = true;
            state.viewport.startX = e.clientX - state.viewport.panX;
            state.viewport.startY = e.clientY - state.viewport.panY;
            elements.viewportStage.classList.add('panning');
        });

        window.addEventListener('mousemove', (e) => {
            // HUD coordinate updates
            if (state.currentIndex >= 0) {
                const stageRect = elements.viewportStage.getBoundingClientRect();
                const curImgX = Math.round((e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom);
                const curImgY = Math.round((e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom);
                elements.coordsHud.textContent = `X: ${curImgX} | Y: ${curImgY}`;
            }

            // Panning
            if (state.viewport.isPanning) {
                state.viewport.panX = e.clientX - state.viewport.startX;
                state.viewport.panY = e.clientY - state.viewport.startY;
                updateTransform();
                return;
            }

            // Moving existing bounding box
            if (state.transformState.isDragging && state.transformState.targetDet) {
                const stageRect = elements.viewportStage.getBoundingClientRect();
                const curImgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
                const curImgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;

                const dx = curImgX - state.transformState.startMouseX;
                const dy = curImgY - state.transformState.startMouseY;

                const [ox1, oy1, ox2, oy2] = state.transformState.originalBox;
                const bw = ox2 - ox1;
                const bh = oy2 - oy1;

                const currentImg = state.images[state.currentIndex];
                const maxW = currentImg ? currentImg.width : 5000;
                const maxH = currentImg ? currentImg.height : 5000;

                const nx1 = Math.max(0, Math.min(ox1 + dx, maxW - bw));
                const ny1 = Math.max(0, Math.min(oy1 + dy, maxH - bh));

                state.transformState.targetDet.box = [
                    Math.round(nx1),
                    Math.round(ny1),
                    Math.round(nx1 + bw),
                    Math.round(ny1 + bh)
                ];

                renderVectorAnnotations();
                return;
            }

            // Resizing existing bounding box via one of 8 handles or 4 side edges
            if (state.transformState.isResizing && state.transformState.targetDet) {
                const stageRect = elements.viewportStage.getBoundingClientRect();
                const curImgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
                const curImgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;

                const dx = curImgX - state.transformState.startMouseX;
                const dy = curImgY - state.transformState.startMouseY;

                const [ox1, oy1, ox2, oy2] = state.transformState.originalBox;
                const handle = state.transformState.resizeHandle;
                const currentImg = state.images[state.currentIndex];
                const maxW = currentImg ? currentImg.width : 5000;
                const maxH = currentImg ? currentImg.height : 5000;
                const minDim = 8;

                let nx1 = ox1, ny1 = oy1, nx2 = ox2, ny2 = oy2;

                if (handle.includes('w')) {
                    nx1 = Math.max(0, Math.min(ox1 + dx, ox2 - minDim));
                }
                if (handle.includes('e')) {
                    nx2 = Math.min(maxW, Math.max(ox2 + dx, ox1 + minDim));
                }
                if (handle.includes('n')) {
                    ny1 = Math.max(0, Math.min(oy1 + dy, oy2 - minDim));
                }
                if (handle.includes('s')) {
                    ny2 = Math.min(maxH, Math.max(oy2 + dy, oy1 + minDim));
                }

                state.transformState.targetDet.box = [
                    Math.round(nx1),
                    Math.round(ny1),
                    Math.round(nx2),
                    Math.round(ny2)
                ];

                renderVectorAnnotations();
                return;
            }

            // Drawing box
            if (state.drawState.isDrawing && state.drawState.tempRect) {
                const stageRect = elements.viewportStage.getBoundingClientRect();
                const imgX = (e.clientX - stageRect.left - state.viewport.panX) / state.viewport.zoom;
                const imgY = (e.clientY - stageRect.top - state.viewport.panY) / state.viewport.zoom;

                const x1 = Math.min(state.drawState.startX, imgX);
                const y1 = Math.min(state.drawState.startY, imgY);
                const w = Math.abs(imgX - state.drawState.startX);
                const h = Math.abs(imgY - state.drawState.startY);

                state.drawState.tempRect.setAttribute('x', x1);
                state.drawState.tempRect.setAttribute('y', y1);
                state.drawState.tempRect.setAttribute('width', w);
                state.drawState.tempRect.setAttribute('height', h);
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (state.viewport.isPanning) {
                state.viewport.isPanning = false;
                elements.viewportStage.classList.remove('panning');
            }

            // Finish moving or resizing
            if (state.transformState.isDragging || state.transformState.isResizing) {
                state.transformState.isDragging = false;
                state.transformState.isResizing = false;
                state.transformState.targetDet = null;
                syncDetectionsToBackend();
                renderDetectionsInspector();
                renderGalleryList();
            }

            if (state.drawState.isDrawing) {
                state.drawState.isDrawing = false;
                if (state.drawState.tempRect && state.currentIndex >= 0) {
                    const rect = state.drawState.tempRect;
                    const x1 = parseFloat(rect.getAttribute('x'));
                    const y1 = parseFloat(rect.getAttribute('y'));
                    const w = parseFloat(rect.getAttribute('width'));
                    const h = parseFloat(rect.getAttribute('height'));
                    rect.remove();
                    state.drawState.tempRect = null;

                    if (w > 10 && h > 10) {
                        const targetQuery = state.activeQueryForDraw || state.queries[0] || 'custom';
                        const queryIdx = state.queries.indexOf(targetQuery);
                        const color = queryIdx >= 0 ? getQueryColor(queryIdx) : '#00f0ff';
                        const newDet = {
                            id: 'box_' + Math.random().toString(36).substr(2, 6),
                            query: targetQuery,
                            color: color,
                            box: [Math.round(x1), Math.round(y1), Math.round(x1 + w), Math.round(y1 + h)],
                            similarity: 1.0,
                            objectness: 1.0,
                            combined_score: 1.0
                        };

                        state.images[state.currentIndex].detections.push(newDet);
                        renderVectorAnnotations();
                        renderDetectionsInspector();
                        renderGalleryList();
                        selectDetection(newDet.id);
                        syncDetectionsToBackend();
                    }
                }
            }
        });

        // Box Modifier Inputs & Listeners
        if (elements.tooltipLabelInput) {
            elements.tooltipLabelInput.addEventListener('change', () => {
                if (!state.selectedDetectionId || state.currentIndex === -1) return;
                const current = state.images[state.currentIndex];
                const det = current.detections.find(d => d.id === state.selectedDetectionId);
                if (det && elements.tooltipLabelInput.value.trim()) {
                    const newLabel = elements.tooltipLabelInput.value.trim().toLowerCase();
                    det.query = newLabel;
                    const qIdx = state.queries.indexOf(newLabel);
                    if (qIdx >= 0) {
                        det.color = getQueryColor(qIdx);
                        if (elements.tooltipColorDot) elements.tooltipColorDot.style.backgroundColor = det.color;
                    }
                    renderVectorAnnotations();
                    renderDetectionsInspector();
                    syncDetectionsToBackend();
                }
            });
            elements.tooltipLabelInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    elements.tooltipLabelInput.blur();
                }
            });
        }

        [elements.boxEditX1, elements.boxEditY1, elements.boxEditW, elements.boxEditH].forEach(inp => {
            if (inp) {
                inp.addEventListener('input', applyManualCoordInputs);
                inp.addEventListener('change', applyManualCoordInputs);
            }
        });

        if (elements.btnSaveBoxEdit) {
            elements.btnSaveBoxEdit.addEventListener('click', () => {
                applyManualCoordInputs();
                elements.btnSaveBoxEdit.textContent = 'Saved ✓';
                setTimeout(() => {
                    elements.btnSaveBoxEdit.textContent = 'Save';
                }, 1200);
            });
        }

        // Tooltip action
        if (elements.tooltipDeleteBtn) {
            elements.tooltipDeleteBtn.addEventListener('click', deleteSelectedDetection);
        }

        // Image pagination navigation
        elements.btnPrevImage.addEventListener('click', () => {
            if (state.currentIndex > 0) setActiveImage(state.currentIndex - 1);
        });
        elements.btnNextImage.addEventListener('click', () => {
            if (state.currentIndex < state.images.length - 1) setActiveImage(state.currentIndex + 1);
        });

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'ArrowRight') {
                if (state.currentIndex < state.images.length - 1) setActiveImage(state.currentIndex + 1);
            } else if (e.key === 'ArrowLeft') {
                if (state.currentIndex > 0) setActiveImage(state.currentIndex - 1);
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                deleteSelectedDetection();
            } else if (e.key === 'Escape') {
                hideTooltip();
            } else if (e.key === 'Enter') {
                runDetectionOnCurrent();
            }
        });

        // Help modal
        elements.btnOpenHelp.addEventListener('click', () => elements.helpModal.classList.remove('hidden'));
        elements.btnCloseHelp.addEventListener('click', () => elements.helpModal.classList.add('hidden'));
        elements.helpModal.addEventListener('click', (e) => {
            if (e.target === elements.helpModal) elements.helpModal.classList.add('hidden');
        });
    }

    // Utility
    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Launch
    init();
});
