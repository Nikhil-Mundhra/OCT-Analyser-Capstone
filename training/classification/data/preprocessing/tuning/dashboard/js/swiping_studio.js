/**
 * swiping_studio.js
 *
 * Interactive Rapid Crop Filtering, Keyboard Swiping, SVG Boundary Vector Editing,
 * and Active-Learning U-Net Retraining Dashboard Engine.
 */

class SwipingStudio {
  constructor() {
    this.queue = [];
    this.currentIndex = 0;
    this.totalScans = 0;
    this.selectedFolder = 'ALL';
    this.viewMode = 'vectors'; // 'vectors', 'crop', 'raw', 'mask'
    // Centralized Interactive Drag State & Multi-Node Selection
    this.activeDrag = null;
    this.selectedNodes = []; // Array of { curveType: 'top'|'bot', idx: number }
    this.stats = { total_curated: 0, total_scans: 0, overall_pct: 0, folders: {} };
    this.retrainPollInterval = null;

    // Swipe touch/mouse drag physics state
    this.isSwipingCard = false;
    this.swipeStartX = 0;
    this.swipeStartY = 0;
    // Background Pre-fetching Cache
    this.preloadedImages = new Map();
    this.isFetchingNextBatch = false;
    this.isAnimating = false;

    this.initElements();
    this.bindEvents();
    this.startBackgroundHeartbeat();
  }

  initElements() {
    this.container = document.getElementById('swiping-studio-view');
    this.card = document.getElementById('swipe-card');
    this.imgRaw = document.getElementById('swipe-img-raw');
    this.svgOverlay = document.getElementById('swipe-svg-overlay');
    this.cropPreviewCanvas = document.getElementById('swipe-crop-canvas');
    this.cleanCropPreview = document.getElementById('swipe-clean-crop-preview');
    this.curatedCountEl = document.getElementById('swipe-curated-count');
    this.progressPctEl = document.getElementById('swipe-progress-pct');
    this.progressBarEl = document.getElementById('swipe-progress-bar-fill');
    this.folderSelect = document.getElementById('swipe-folder-select');
    this.sampleMetaEl = document.getElementById('swipe-sample-meta');
    this.stampApprove = document.getElementById('stamp-approve');
    this.stampSkip = document.getElementById('stamp-skip');
    this.viewModeBadge = document.getElementById('swipe-view-mode-badge');

    // Retrain Button and Modal elements
    this.btnTriggerRetrain = document.getElementById('btn-trigger-retrain');
    this.btnRetrainFill = document.getElementById('btn-retrain-fill');
    this.btnRetrainShimmer = document.getElementById('btn-retrain-shimmer');
    this.btnRetrainText = document.getElementById('btn-retrain-text');
    this.retrainModal = document.getElementById('retrain-modal');
    this.retrainStatusCard = document.getElementById('retrain-status-card');
    this.retrainProgressBar = document.getElementById('retrain-progress-bar-fill');
    this.retrainPctBadge = document.getElementById('retrain-pct-badge');
    this.retrainMsgEl = document.getElementById('retrain-msg');
    this.retrainMetricsEl = document.getElementById('retrain-metrics');
    this.btnStartRetrainExec = document.getElementById('btn-start-retrain-exec');
    this.btnStopRetrainExec = document.getElementById('btn-stop-retrain-exec');

    // Model History elements
    this.btnModelHistory = document.getElementById('btn-model-history');
    this.modalModelHistory = document.getElementById('modal-model-history');
    this.btnCloseHistoryModal = document.getElementById('btn-close-history-modal');
    this.historyRoundsList = document.getElementById('history-rounds-list');
    this.historyStatDice = document.getElementById('history-stat-dice');
    this.historyStatIou = document.getElementById('history-stat-iou');
    this.historyStatRounds = document.getElementById('history-stat-rounds');
    this.historyStatCurated = document.getElementById('history-stat-curated');
    this.historyTotalRoundsBadge = document.getElementById('history-total-rounds-badge');

    this.backgroundHeartbeatInterval = null;
  }

  bindEvents() {
    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (!this.isActiveView()) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if (e.key === 'ArrowRight' || e.code === 'KeyD') {
        e.preventDefault();
        this.approveCurrentCrop();
      } else if (e.key === 'ArrowLeft' || e.code === 'KeyA') {
        e.preventDefault();
        this.skipCurrentCrop();
      } else if (e.code === 'Space' || e.key === 'ArrowUp' || e.code === 'KeyV') {
        e.preventDefault();
        this.toggleViewMode();
      } else if (e.key === 'ArrowDown' || e.code === 'KeyZ') {
        e.preventDefault();
        this.stepPrevious();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (this.selectedNodes && this.selectedNodes.length > 0) {
          e.preventDefault();
          this.deleteSelectedNodes();
        }
      } else if (e.code === 'KeyR' && e.shiftKey) {
        e.preventDefault();
        this.openRetrainModal();
      } else if (e.code === 'KeyH' && e.shiftKey) {
        e.preventDefault();
        this.openHistoryModal();
      }
    });

    // Folder Filter Change
    if (this.folderSelect) {
      this.folderSelect.addEventListener('change', (e) => {
        this.selectedFolder = e.target.value;
        this.loadQueue(0);
      });
    }

    // Action Buttons
    const btnApprove = document.getElementById('btn-swipe-approve');
    const btnSkip = document.getElementById('btn-swipe-skip');
    const btnToggleView = document.getElementById('btn-swipe-toggle-view');
    const btnRetrain = document.getElementById('btn-trigger-retrain');
    const btnCloseModal = document.getElementById('btn-close-retrain-modal');
    const btnStartRetrain = document.getElementById('btn-start-retrain-exec');
    const btnStopRetrain = document.getElementById('btn-stop-retrain-exec');

    if (btnApprove) btnApprove.addEventListener('click', () => this.approveCurrentCrop());
    if (btnSkip) btnSkip.addEventListener('click', () => this.skipCurrentCrop());
    if (btnToggleView) btnToggleView.addEventListener('click', () => this.toggleViewMode());
    if (btnRetrain) btnRetrain.addEventListener('click', () => this.openRetrainModal());
    if (this.headerRetrainBadge) this.headerRetrainBadge.addEventListener('click', () => this.openRetrainModal());
    if (btnCloseModal) btnCloseModal.addEventListener('click', () => this.closeRetrainModal());
    if (btnStartRetrain) btnStartRetrain.addEventListener('click', () => this.startRetraining());
    if (btnStopRetrain) btnStopRetrain.addEventListener('click', () => this.stopRetraining());

    // History Modal Buttons
    if (this.btnModelHistory) this.btnModelHistory.addEventListener('click', () => this.openHistoryModal());
    if (this.btnCloseHistoryModal) this.btnCloseHistoryModal.addEventListener('click', () => this.closeHistoryModal());

    // Unified Global Drag Controller (Registered once)
    window.addEventListener('mousemove', (e) => this.handleGlobalDragMove(e));
    window.addEventListener('mouseup', (e) => this.handleGlobalDragEnd(e));
    window.addEventListener('touchmove', (e) => this.handleGlobalDragMove(e), { passive: false });
    window.addEventListener('touchend', (e) => this.handleGlobalDragEnd(e));

    // Mouse / Touch Swiping on Card
    if (this.card) {
      const onDragStart = (e) => {
        if (this.activeDrag) return;
        if (e.target.closest('#swipe-svg-overlay') || e.target.tagName === 'circle' || e.target.tagName === 'path' || e.target.tagName === 'rect' || e.target.tagName === 'line') return;
        this.isSwipingCard = true;
        this.swipeStartX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        this.swipeStartY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
        this.card.style.transition = 'none';
      };

      const onDragMove = (e) => {
        if (!this.isSwipingCard || this.activeDrag) return;
        const currentX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        const deltaX = currentX - this.swipeStartX;
        this.currentTranslateX = deltaX;

        const rotation = deltaX * 0.05;
        this.card.style.transform = `translateX(${deltaX}px) rotate(${rotation}deg)`;

        // Update stamp opacities
        if (deltaX > 40) {
          const opacity = Math.min(1, (deltaX - 40) / 100);
          if (this.stampApprove) this.stampApprove.style.opacity = opacity;
          if (this.stampSkip) this.stampSkip.style.opacity = 0;
        } else if (deltaX < -40) {
          const opacity = Math.min(1, (-deltaX - 40) / 100);
          if (this.stampSkip) this.stampSkip.style.opacity = opacity;
          if (this.stampApprove) this.stampApprove.style.opacity = 0;
        } else {
          if (this.stampApprove) this.stampApprove.style.opacity = 0;
          if (this.stampSkip) this.stampSkip.style.opacity = 0;
        }
      };

      const onDragEnd = () => {
        if (!this.isSwipingCard) return;
        this.isSwipingCard = false;
        this.card.style.transition = 'transform 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)';

        if (this.currentTranslateX > 110) {
          this.card.style.transform = 'translateX(600px) rotate(20deg)';
          setTimeout(() => this.approveCurrentCrop(), 180);
        } else if (this.currentTranslateX < -110) {
          this.card.style.transform = 'translateX(-600px) rotate(-20deg)';
          setTimeout(() => this.skipCurrentCrop(), 180);
        } else {
          this.card.style.transform = 'translateX(0px) rotate(0deg)';
          if (this.stampApprove) this.stampApprove.style.opacity = 0;
          if (this.stampSkip) this.stampSkip.style.opacity = 0;
        }
        this.currentTranslateX = 0;
      };

      this.card.addEventListener('mousedown', onDragStart);
      this.card.addEventListener('touchstart', onDragStart, { passive: true });
    }
  }

  isActiveView() {
    return this.container && this.container.style.display !== 'none';
  }

  async activate() {
    if (this.container) this.container.style.display = 'flex';
    const tuningApp = document.querySelector('.app-container');
    if (tuningApp) tuningApp.style.display = 'none';
    await this.refreshStats();
    await this.populateFolders();
    await this.loadQueue(0);
  }

  deactivate() {
    if (this.container) this.container.style.display = 'none';
    const tuningApp = document.querySelector('.app-container');
    if (tuningApp) tuningApp.style.display = 'grid';
  }

  scheduleRefreshStats(delay = 800) {
    if (this._statsTimeout) clearTimeout(this._statsTimeout);
    this._statsTimeout = setTimeout(() => {
      this.refreshStats();
    }, delay);
  }

  async refreshStats() {
    try {
      this.stats = await fetchCurationStats();
      let cur = this.stats.total_curated;
      let tot = this.stats.total_scans;
      let pct = this.stats.overall_pct;

      if (this.selectedFolder && this.selectedFolder !== 'ALL' && this.stats.folders && this.stats.folders[this.selectedFolder]) {
        const fInfo = this.stats.folders[this.selectedFolder];
        cur = fInfo.curated;
        tot = fInfo.total;
        pct = fInfo.pct;
      }

      if (this.curatedCountEl) this.curatedCountEl.textContent = `${cur} / ${tot}`;
      if (this.progressPctEl) this.progressPctEl.textContent = `${pct.toFixed(1)}%`;
      if (this.progressBarEl) this.progressBarEl.style.width = `${Math.min(100, pct)}%`;
      this.updateRetrainDatasetStats();
    } catch (e) {
      console.warn('Failed to refresh stats:', e);
    }
  }

  async populateFolders() {
    if (!this.folderSelect) return;
    const currentVal = this.folderSelect.value || 'ALL';
    this.folderSelect.innerHTML = '<option value="ALL">ALL DISEASES (Complete Queue)</option>';

    if (this.stats && this.stats.folders) {
      Object.keys(this.stats.folders).sort().forEach(folder => {
        const fInfo = this.stats.folders[folder];
        const opt = document.createElement('option');
        opt.value = folder;
        opt.textContent = `${folder} (${fInfo.curated}/${fInfo.total} &mdash; ${fInfo.pct.toFixed(0)}%)`;
        this.folderSelect.appendChild(opt);
      });
    }
    this.folderSelect.value = currentVal;
  }

  onFolderChange(folder) {
    this.selectedFolder = folder || 'ALL';
    if (this.folderSelect) this.folderSelect.value = this.selectedFolder;
    this.loadQueue(0);
  }

  async loadQueue(offset = 0, clearCache = false) {
    try {
      if (this.sampleMetaEl) this.sampleMetaEl.textContent = 'Loading scans with Attention U-Net inferences...';
      const res = await fetchCropFilterQueue(this.selectedFolder, offset, 40, clearCache);
      this.queue = res.items || [];
      this.totalScans = res.total || 0;
      this.currentIndex = 0;

      if (this.queue.length > 0) {
        this.renderCurrentScan();
      } else {
        this.renderEmptyQueue();
      }
    } catch (e) {
      console.error('Failed to load crop filter queue:', e);
      if (this.sampleMetaEl) this.sampleMetaEl.textContent = 'Error loading scans from Classified/ dataset.';
    }
  }

  renderCurrentScan() {
    if (!this.card) return;
    this.card.style.transition = 'none';
    this.card.style.transform = 'translateX(0px) rotate(0deg)';
    if (this.stampApprove) this.stampApprove.style.opacity = 0;
    if (this.stampSkip) this.stampSkip.style.opacity = 0;

    const item = this.queue[this.currentIndex];
    if (!item) {
      this.loadQueue(this.currentIndex);
      return;
    }

    if (this.sampleMetaEl) {
      const curBadge = item.is_curated ? '<span class="badge" style="background:#10b981;color:#fff;">VERIFIED GROUND TRUTH</span>' : '<span class="badge" style="background:#3b82f6;color:#fff;">UNET PREDICTED</span>';
      const scanNum = this.currentIndex + 1;
      const totalNum = this.totalScans || this.queue.length;
      this.sampleMetaEl.innerHTML = `<span style="color:#38bdf8;font-weight:700;margin-right:8px;">[Scan ${scanNum} of ${totalNum}]</span><strong>${item.folder}</strong> / ${item.filename} &bull; ${item.width}&times;${item.height}px ${curBadge}`;
    }

    // Immediately trigger lookahead image preloading for next 5 scans
    this.preloadNextImages(5);

    // Clear any prior node selection on scan switch
    this.clearSelection();

    if (this.imgRaw) {
      this.imgRaw.crossOrigin = "anonymous";
      this.imgRaw.onload = () => {
        this.renderVectorsAndCrop(item);
      };
      this.imgRaw.onerror = () => {
        console.error("Failed to load image:", item.image_url);
        if (this.sampleMetaEl) {
          this.sampleMetaEl.innerHTML = `<span style="color:#ef4444;font-weight:700;">Error loading scan:</span> ${item.folder} / ${item.filename}`;
        }
      };

      // Check if already in browser/DOM memory
      if (this.imgRaw.src.endsWith(item.image_url) && this.imgRaw.complete && this.imgRaw.naturalWidth > 0) {
        this.renderVectorsAndCrop(item);
      } else {
        this.imgRaw.src = item.image_url;
      }
    }
  }

  preloadNextImages(count = 12) {
    for (let i = this.currentIndex + 1; i <= Math.min(this.currentIndex + count, this.queue.length - 1); i++) {
      const nextItem = this.queue[i];
      if (nextItem && !this.preloadedImages.has(nextItem.image_url)) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = nextItem.image_url;
        this.preloadedImages.set(nextItem.image_url, img);
      }
    }

    // Background prefetch next queue slice well before reaching end
    if (this.currentIndex + 18 >= this.queue.length && !this.isFetchingNextBatch && this.queue.length < this.totalScans) {
      this.prefetchNextBatch();
    }
  }

  async prefetchNextBatch() {
    this.isFetchingNextBatch = true;
    try {
      const res = await fetchCropFilterQueue(this.selectedFolder, this.queue.length, 30);
      if (res && res.items && res.items.length > 0) {
        this.queue = this.queue.concat(res.items);
        this.preloadNextImages(10);
      }
    } catch (e) {
      console.warn('Prefetch error:', e);
    } finally {
      this.isFetchingNextBatch = false;
    }
  }

  clientToSvg(clientX, clientY) {
    if (!this.svgOverlay) return { x: 0, y: 0 };
    const pt = this.svgOverlay.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = this.svgOverlay.getScreenCTM();
    if (ctm) {
      const svgPt = pt.matrixTransform(ctm.inverse());
      return { x: svgPt.x, y: svgPt.y };
    }
    const rect = this.svgOverlay.getBoundingClientRect();
    const item = this.queue[this.currentIndex];
    const w = item ? item.width : (this.svgOverlay.viewBox.baseVal.width || rect.width);
    const h = item ? item.height : (this.svgOverlay.viewBox.baseVal.height || rect.height);
    return {
      x: ((clientX - rect.left) / rect.width) * w,
      y: ((clientY - rect.top) / rect.height) * h
    };
  }

  computeMonotoneHermiteBezierSegments(points) {
    const n = points ? points.length : 0;
    if (n < 2) return [];

    // Step 1: Secant slopes between adjacent points
    const dx = new Array(n - 1);
    const dy = new Array(n - 1);
    const slopes = new Array(n - 1);

    for (let i = 0; i < n - 1; i++) {
      dx[i] = Math.max(1e-4, points[i + 1].x - points[i].x);
      dy[i] = points[i + 1].y - points[i].y;
      slopes[i] = dy[i] / dx[i];
    }

    // Step 2: Monotone PCHIP tangents at each node to strictly eliminate overshoot & ringing
    const tangents = new Array(n);
    tangents[0] = slopes[0];
    tangents[n - 1] = slopes[n - 2];

    for (let i = 1; i < n - 1; i++) {
      const s0 = slopes[i - 1];
      const s1 = slopes[i];
      if (s0 * s1 <= 0) {
        // Local extremum -> Flat slope strictly prevents curve ballooning
        tangents[i] = 0;
      } else {
        // Weighted harmonic mean preserving monotonic curvature
        const w0 = 2 * dx[i] + dx[i - 1];
        const w1 = dx[i] + 2 * dx[i - 1];
        tangents[i] = (w0 + w1) / (w0 / s0 + w1 / s1);
      }
    }

    // Step 3: Cubic Bézier control points
    const segments = [];
    for (let i = 0; i < n - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];
      const h = dx[i];
      const cp1 = {
        x: p1.x + h / 3,
        y: p1.y + tangents[i] * (h / 3)
      };
      const cp2 = {
        x: p2.x - h / 3,
        y: p2.y - tangents[i + 1] * (h / 3)
      };
      segments.push({ p1, cp1, cp2, p2 });
    }
    return segments;
  }

  pointsToSmoothPathD(points) {
    if (!points || points.length === 0) return '';
    if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
    const segments = this.computeMonotoneHermiteBezierSegments(points);
    let d = `M ${points[0].x} ${points[0].y}`;
    for (const seg of segments) {
      d += ` C ${seg.cp1.x.toFixed(2)} ${seg.cp1.y.toFixed(2)}, ${seg.cp2.x.toFixed(2)} ${seg.cp2.y.toFixed(2)}, ${seg.p2.x} ${seg.p2.y}`;
    }
    return d;
  }

  pointsToSmoothBandPolygonD(topPts, botPts) {
    if (!topPts || !botPts || topPts.length === 0 || botPts.length === 0) return '';
    const topD = this.pointsToSmoothPathD(topPts);
    const revBot = [...botPts].reverse();
    const botSegments = this.computeMonotoneHermiteBezierSegments(revBot);
    let botSegD = '';
    for (const seg of botSegments) {
      botSegD += ` C ${seg.cp1.x.toFixed(2)} ${seg.cp1.y.toFixed(2)}, ${seg.cp2.x.toFixed(2)} ${seg.cp2.y.toFixed(2)}, ${seg.p2.x} ${seg.p2.y}`;
    }
    return `${topD} L ${revBot[0].x} ${revBot[0].y} ${botSegD} Z`;
  }

  traceSmoothPathToCanvas(ctx, topPts, botPts) {
    if (!topPts || !botPts || topPts.length === 0 || botPts.length === 0) return;
    const topSegments = this.computeMonotoneHermiteBezierSegments(topPts);
    const revBot = [...botPts].reverse();
    const botSegments = this.computeMonotoneHermiteBezierSegments(revBot);

    ctx.beginPath();
    ctx.moveTo(topPts[0].x, topPts[0].y);
    for (const seg of topSegments) {
      ctx.bezierCurveTo(seg.cp1.x, seg.cp1.y, seg.cp2.x, seg.cp2.y, seg.p2.x, seg.p2.y);
    }
    ctx.lineTo(revBot[0].x, revBot[0].y);
    for (const seg of botSegments) {
      ctx.bezierCurveTo(seg.cp1.x, seg.cp1.y, seg.cp2.x, seg.cp2.y, seg.p2.x, seg.p2.y);
    }
    ctx.closePath();
  }

  getCentralPeakY(pts, w, mode) {
    if (!pts || pts.length === 0) return 0;
    const midX = w / 2;
    const halfSpan = 90; // Generous span covering 84px capsule + surrounding anatomical slope
    const centralPts = pts.filter(p => Math.abs(p.x - midX) <= halfSpan);
    const targetPts = centralPts.length > 0 ? centralPts : pts;

    if (mode === 'top') {
      // Top peak = minimum Y (highest anatomical point towards top of image)
      return Math.min(...targetPts.map(p => p.y));
    } else {
      // Bottom peak = maximum Y (lowest anatomical point towards bottom of image)
      return Math.max(...targetPts.map(p => p.y));
    }
  }

  renderVectorsAndCrop(item) {
    if (!this.svgOverlay || !item) return;

    const w = item.width;
    const h = item.height;
    const topPts = item.y_top_points || [];
    const botPts = item.y_bot_points || [];
    if (item.crop_left_top === undefined) item.crop_left_top = (item.crop_left !== undefined ? item.crop_left : 0);
    if (item.crop_left_bot === undefined) item.crop_left_bot = (item.crop_left !== undefined ? item.crop_left : 0);
    if (item.crop_right_top === undefined) item.crop_right_top = (item.crop_right !== undefined ? item.crop_right : w);
    if (item.crop_right_bot === undefined) item.crop_right_bot = (item.crop_right !== undefined ? item.crop_right : w);
    item.crop_left = Math.round((item.crop_left_top + item.crop_left_bot) / 2);
    item.crop_right = Math.round((item.crop_right_top + item.crop_right_bot) / 2);

    this.svgOverlay.setAttribute('viewBox', `0 0 ${w} ${h}`);
    this.svgOverlay.innerHTML = '';

    // Marquee Selection Drag on Empty Background Area with Pixel-Perfect Matrix Transformation
    const onSvgBgStart = (e) => {
      if (e.target.tagName === 'circle' || e.target.id === 'svg-hit-left' || e.target.id === 'svg-hit-right' || e.target.id === 'svg-top-hit' || e.target.id === 'svg-bot-hit' || e.target.id === 'svg-retinal-band') {
        return;
      }
      const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
      const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
      const startSvg = this.clientToSvg(clientX, clientY);
      this.activeDrag = {
        type: 'marquee_select',
        item,
        startSvgX: startSvg.x,
        startSvgY: startSvg.y,
        startClientX: clientX,
        startClientY: clientY,
        moved: false
      };
    };
    this.svgOverlay.onmousedown = onSvgBgStart;
    this.svgOverlay.ontouchstart = onSvgBgStart;

    // Dynamic Proximity Node Hover Engine: Illuminates exact closest node + 2 adjacent neighbors
    this.svgOverlay.onmousemove = (e) => {
      this.updateProximityNodeHover(e.clientX, e.clientY, item);
    };
    this.svgOverlay.onmouseleave = () => {
      this.clearProximityNodeHover();
    };

    // 1. Render Left & Right Lateral Curtains and Guides (Base Layer)
    this.renderLateralCurtains(item, w, h);

    if (topPts.length > 0 && botPts.length > 0) {
      // 2. Shaded Retinal Band Polygon (Draggable Whole Tissue) - Smooth Spline Contour
      const polyD = this.pointsToSmoothBandPolygonD(topPts, botPts);

      const polyEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      polyEl.setAttribute('d', polyD);
      polyEl.setAttribute('fill', 'rgba(0, 242, 254, 0.16)');
      polyEl.setAttribute('stroke', 'none');
      polyEl.setAttribute('class', 'draggable-tissue-band');
      polyEl.style.cursor = 'move';
      polyEl.id = 'svg-retinal-band';
      this.bindCurveGlobalDrag(polyEl, item, 'both');
      this.svgOverlay.appendChild(polyEl);

      // 3. Top Boundary Path (Cyan ILM) - Smooth Spline Contour
      const topD = this.pointsToSmoothPathD(topPts);

      const topPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      topPath.setAttribute('d', topD);
      topPath.setAttribute('stroke', '#00f2fe');
      topPath.setAttribute('stroke-width', '2.5');
      topPath.setAttribute('fill', 'none');
      topPath.id = 'svg-top-path';
      this.svgOverlay.appendChild(topPath);

      const topHit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      topHit.setAttribute('d', topD);
      topHit.setAttribute('stroke', 'transparent');
      topHit.setAttribute('stroke-width', '44');
      topHit.setAttribute('fill', 'none');
      topHit.style.cursor = 'ns-resize';
      topHit.id = 'svg-top-hit';
      this.bindCurveGlobalDrag(topHit, item, 'top');

      // Double-click top curve to select all top nodes
      topHit.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.selectedNodes = topPts.map((_, idx) => ({ curveType: 'top', idx }));
        this.updateSelectedNodesVisual();
      };

      // Right-Click on Top Curve to insert a node between existing points
      const onTopRightClick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
        const svgPt = this.clientToSvg(clientX, clientY);
        this.insertNodeOnCurve(item, 'top', Math.round(svgPt.x), Math.round(svgPt.y));
      };
      topHit.oncontextmenu = onTopRightClick;
      topPath.oncontextmenu = onTopRightClick;
      this.svgOverlay.appendChild(topHit);

      // 4. Bottom Boundary Path (Orange Choroid CSI) - Smooth Spline Contour
      const botD = this.pointsToSmoothPathD(botPts);

      const botPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      botPath.setAttribute('d', botD);
      botPath.setAttribute('stroke', '#ff9100');
      botPath.setAttribute('stroke-width', '2.5');
      botPath.setAttribute('fill', 'none');
      botPath.id = 'svg-bot-path';
      this.svgOverlay.appendChild(botPath);

      const botHit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      botHit.setAttribute('d', botD);
      botHit.setAttribute('stroke', 'transparent');
      botHit.setAttribute('stroke-width', '44');
      botHit.setAttribute('fill', 'none');
      botHit.style.cursor = 'ns-resize';
      botHit.id = 'svg-bot-hit';
      this.bindCurveGlobalDrag(botHit, item, 'bot');

      // Double-click bottom curve to select all bottom nodes
      botHit.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.selectedNodes = botPts.map((_, idx) => ({ curveType: 'bot', idx }));
        this.updateSelectedNodesVisual();
      };

      // Right-Click on Bottom Curve to insert a node between existing points
      const onBotRightClick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
        const svgPt = this.clientToSvg(clientX, clientY);
        this.insertNodeOnCurve(item, 'bot', Math.round(svgPt.x), Math.round(svgPt.y));
      };
      botHit.oncontextmenu = onBotRightClick;
      botPath.oncontextmenu = onBotRightClick;
      this.svgOverlay.appendChild(botHit);

      // 5. Interactive Control Point Handles (TOP LAYER - Highest Click Priority)
      const handlesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      handlesGroup.id = 'svg-handles-group';

      topPts.forEach((pt, idx) => {
        // Individual non-overlapping hit target (r=8 for 48-node high-density mode)
        const hitTarget = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        hitTarget.setAttribute('cx', pt.x);
        hitTarget.setAttribute('cy', pt.y);
        hitTarget.setAttribute('r', '8');
        hitTarget.setAttribute('fill', 'transparent');
        hitTarget.setAttribute('data-curve', 'top');
        hitTarget.setAttribute('data-idx', idx);
        hitTarget.style.cursor = 'grab';
        this.bindPointDrag(hitTarget, item, 'top', idx);
        // Right-click on node to delete it
        hitTarget.oncontextmenu = (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.removeNodeFromCurve(item, 'top', idx);
        };
        handlesGroup.appendChild(hitTarget);

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pt.x);
        circle.setAttribute('cy', pt.y);
        circle.setAttribute('r', this.isSelected('top', idx) ? '4.0' : '2.75');
        circle.setAttribute('fill', '#00f2fe');
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '1.0');
        circle.setAttribute('class', `vector-handle top-handle${this.isSelected('top', idx) ? ' node-selected' : ''}`);
        circle.setAttribute('data-curve', 'top');
        circle.setAttribute('data-idx', idx);
        circle.style.pointerEvents = 'none'; // hitTarget handles the drag

        handlesGroup.appendChild(circle);
      });

      botPts.forEach((pt, idx) => {
        const hitTarget = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        hitTarget.setAttribute('cx', pt.x);
        hitTarget.setAttribute('cy', pt.y);
        hitTarget.setAttribute('r', '8');
        hitTarget.setAttribute('fill', 'transparent');
        hitTarget.setAttribute('data-curve', 'bot');
        hitTarget.setAttribute('data-idx', idx);
        hitTarget.style.cursor = 'grab';
        this.bindPointDrag(hitTarget, item, 'bot', idx);
        // Right-click on node to delete it
        hitTarget.oncontextmenu = (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.removeNodeFromCurve(item, 'bot', idx);
        };
        handlesGroup.appendChild(hitTarget);

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pt.x);
        circle.setAttribute('cy', pt.y);
        circle.setAttribute('r', this.isSelected('bot', idx) ? '4.0' : '2.75');
        circle.setAttribute('fill', '#ff9100');
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '1.0');
        circle.setAttribute('class', `vector-handle bot-handle${this.isSelected('bot', idx) ? ' node-selected' : ''}`);
        circle.setAttribute('data-curve', 'bot');
        circle.setAttribute('data-idx', idx);
        circle.style.pointerEvents = 'none'; // hitTarget handles the drag

        handlesGroup.appendChild(circle);
      });

      this.svgOverlay.appendChild(handlesGroup);

      // 6. Dedicated Minimalist Central Curve Grab Handles (Central Neighborhood Peak Calculation)
      const topPeakY = this.getCentralPeakY(topPts, w, 'top');
      const botPeakY = this.getCentralPeakY(botPts, w, 'bot');

      // Top Line Grab Handle Group (Hover-enabled Cyan Capsule, shifted up by 48px from central peak)
      const topGrabG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      topGrabG.id = 'svg-top-grab-handle';
      topGrabG.setAttribute('class', 'curve-line-handle curve-line-handle-top');
      topGrabG.setAttribute('transform', `translate(${w / 2}, ${Math.max(12, topPeakY - 48)})`);

      const topHitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      topHitRect.setAttribute('x', '-70');
      topHitRect.setAttribute('y', '-24');
      topHitRect.setAttribute('width', '140');
      topHitRect.setAttribute('height', '48');
      topHitRect.setAttribute('fill', 'transparent');
      topGrabG.appendChild(topHitRect);

      const topVisual = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      topVisual.setAttribute('x', '-42');
      topVisual.setAttribute('y', '-5');
      topVisual.setAttribute('width', '84');
      topVisual.setAttribute('height', '10');
      topVisual.setAttribute('rx', '5');
      topVisual.setAttribute('ry', '5');
      topVisual.setAttribute('fill', '#00f2fe');
      topVisual.setAttribute('stroke', '#ffffff');
      topVisual.setAttribute('stroke-width', '1.4');
      topVisual.setAttribute('class', 'handle-visual');
      topGrabG.appendChild(topVisual);

      this.bindCurveGlobalDrag(topGrabG, item, 'top');
      this.svgOverlay.appendChild(topGrabG);

      // Bottom Line Grab Handle Group (Hover-enabled Orange Capsule, shifted down by 48px from central peak)
      const botGrabG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      botGrabG.id = 'svg-bot-grab-handle';
      botGrabG.setAttribute('class', 'curve-line-handle curve-line-handle-bot');
      botGrabG.setAttribute('transform', `translate(${w / 2}, ${Math.min(h - 12, botPeakY + 48)})`);

      const botHitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      botHitRect.setAttribute('x', '-70');
      botHitRect.setAttribute('y', '-24');
      botHitRect.setAttribute('width', '140');
      botHitRect.setAttribute('height', '48');
      botHitRect.setAttribute('fill', 'transparent');
      botGrabG.appendChild(botHitRect);

      const botVisual = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      botVisual.setAttribute('x', '-42');
      botVisual.setAttribute('y', '-5');
      botVisual.setAttribute('width', '84');
      botVisual.setAttribute('height', '10');
      botVisual.setAttribute('rx', '5');
      botVisual.setAttribute('ry', '5');
      botVisual.setAttribute('fill', '#ff9100');
      botVisual.setAttribute('stroke', '#ffffff');
      botVisual.setAttribute('stroke-width', '1.4');
      botVisual.setAttribute('class', 'handle-visual');
      botGrabG.appendChild(botVisual);

      this.bindCurveGlobalDrag(botGrabG, item, 'bot');
      this.svgOverlay.appendChild(botGrabG);
    }

    this.renderLiveCropCanvas(item);
  }

  renderLateralCurtains(item, w, h) {
    const clt = item.crop_left_top !== undefined ? item.crop_left_top : (item.crop_left || 0);
    const clb = item.crop_left_bot !== undefined ? item.crop_left_bot : (item.crop_left || 0);
    const crt = item.crop_right_top !== undefined ? item.crop_right_top : (item.crop_right || w);
    const crb = item.crop_right_bot !== undefined ? item.crop_right_bot : (item.crop_right || w);

    // 1. Left Curtain Polygon (Darkened uncropped region)
    const curtainLeft = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    curtainLeft.setAttribute('points', `0,0 ${clt},0 ${clb},${h} 0,${h}`);
    curtainLeft.setAttribute('fill', 'rgba(0, 0, 0, 0.68)');
    curtainLeft.id = 'svg-curtain-left';
    this.svgOverlay.appendChild(curtainLeft);

    // 2. Left Slanted Guide Line
    const lineLeft = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineLeft.setAttribute('x1', clt);
    lineLeft.setAttribute('y1', 0);
    lineLeft.setAttribute('x2', clb);
    lineLeft.setAttribute('y2', h);
    lineLeft.setAttribute('stroke', '#00f2fe');
    lineLeft.setAttribute('stroke-width', '2.5');
    lineLeft.setAttribute('stroke-dasharray', '6 4');
    lineLeft.id = 'svg-line-left';
    this.svgOverlay.appendChild(lineLeft);

    // 3. Left Line Body Hit Area (for dragging the entire slanted line horizontally)
    const hitLeft = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hitLeft.setAttribute('x1', clt);
    hitLeft.setAttribute('y1', 0);
    hitLeft.setAttribute('x2', clb);
    hitLeft.setAttribute('y2', h);
    hitLeft.setAttribute('stroke', 'transparent');
    hitLeft.setAttribute('stroke-width', '36');
    hitLeft.style.cursor = 'ew-resize';
    hitLeft.id = 'svg-hit-left';
    this.bindLateralDrag(hitLeft, item, 'left_line');
    this.svgOverlay.appendChild(hitLeft);

    // 4. Left Top Corner Handle (for rotating/slanting top-left endpoint)
    const handleLeftTop = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    handleLeftTop.setAttribute('cx', clt);
    handleLeftTop.setAttribute('cy', 8);
    handleLeftTop.setAttribute('r', '4');
    handleLeftTop.setAttribute('fill', '#00f2fe');
    handleLeftTop.setAttribute('stroke', '#ffffff');
    handleLeftTop.setAttribute('stroke-width', '1.2');
    handleLeftTop.setAttribute('class', 'lateral-corner-handle');
    handleLeftTop.id = 'svg-handle-left-top';
    this.svgOverlay.appendChild(handleLeftTop);

    const hitLeftTop = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    hitLeftTop.setAttribute('cx', clt);
    hitLeftTop.setAttribute('cy', 8);
    hitLeftTop.setAttribute('r', '14');
    hitLeftTop.setAttribute('fill', 'transparent');
    hitLeftTop.style.cursor = 'ew-resize';
    hitLeftTop.id = 'svg-hit-left-top';
    this.bindLateralDrag(hitLeftTop, item, 'left_top');
    this.svgOverlay.appendChild(hitLeftTop);

    // 5. Left Bottom Corner Handle (for rotating/slanting bottom-left endpoint)
    const handleLeftBot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    handleLeftBot.setAttribute('cx', clb);
    handleLeftBot.setAttribute('cy', h - 8);
    handleLeftBot.setAttribute('r', '4');
    handleLeftBot.setAttribute('fill', '#00f2fe');
    handleLeftBot.setAttribute('stroke', '#ffffff');
    handleLeftBot.setAttribute('stroke-width', '1.2');
    handleLeftBot.setAttribute('class', 'lateral-corner-handle');
    handleLeftBot.id = 'svg-handle-left-bot';
    this.svgOverlay.appendChild(handleLeftBot);

    const hitLeftBot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    hitLeftBot.setAttribute('cx', clb);
    hitLeftBot.setAttribute('cy', h - 8);
    hitLeftBot.setAttribute('r', '14');
    hitLeftBot.setAttribute('fill', 'transparent');
    hitLeftBot.style.cursor = 'ew-resize';
    hitLeftBot.id = 'svg-hit-left-bot';
    this.bindLateralDrag(hitLeftBot, item, 'left_bot');
    this.svgOverlay.appendChild(hitLeftBot);

    // 6. Right Curtain Polygon (Darkened uncropped region)
    const curtainRight = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    curtainRight.setAttribute('points', `${crt},0 ${w},0 ${w},${h} ${crb},${h}`);
    curtainRight.setAttribute('fill', 'rgba(0, 0, 0, 0.68)');
    curtainRight.id = 'svg-curtain-right';
    this.svgOverlay.appendChild(curtainRight);

    // 7. Right Slanted Guide Line
    const lineRight = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineRight.setAttribute('x1', crt);
    lineRight.setAttribute('y1', 0);
    lineRight.setAttribute('x2', crb);
    lineRight.setAttribute('y2', h);
    lineRight.setAttribute('stroke', '#ff9100');
    lineRight.setAttribute('stroke-width', '2.5');
    lineRight.setAttribute('stroke-dasharray', '6 4');
    lineRight.id = 'svg-line-right';
    this.svgOverlay.appendChild(lineRight);

    // 8. Right Line Body Hit Area
    const hitRight = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hitRight.setAttribute('x1', crt);
    hitRight.setAttribute('y1', 0);
    hitRight.setAttribute('x2', crb);
    hitRight.setAttribute('y2', h);
    hitRight.setAttribute('stroke', 'transparent');
    hitRight.setAttribute('stroke-width', '36');
    hitRight.style.cursor = 'ew-resize';
    hitRight.id = 'svg-hit-right';
    this.bindLateralDrag(hitRight, item, 'right_line');
    this.svgOverlay.appendChild(hitRight);

    // 9. Right Top Corner Handle (for rotating/slanting top-right endpoint)
    const handleRightTop = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    handleRightTop.setAttribute('cx', crt);
    handleRightTop.setAttribute('cy', 8);
    handleRightTop.setAttribute('r', '4');
    handleRightTop.setAttribute('fill', '#ff9100');
    handleRightTop.setAttribute('stroke', '#ffffff');
    handleRightTop.setAttribute('stroke-width', '1.2');
    handleRightTop.setAttribute('class', 'lateral-corner-handle');
    handleRightTop.id = 'svg-handle-right-top';
    this.svgOverlay.appendChild(handleRightTop);

    const hitRightTop = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    hitRightTop.setAttribute('cx', crt);
    hitRightTop.setAttribute('cy', 8);
    hitRightTop.setAttribute('r', '14');
    hitRightTop.setAttribute('fill', 'transparent');
    hitRightTop.style.cursor = 'ew-resize';
    hitRightTop.id = 'svg-hit-right-top';
    this.bindLateralDrag(hitRightTop, item, 'right_top');
    this.svgOverlay.appendChild(hitRightTop);

    // 10. Right Bottom Corner Handle (for rotating/slanting bottom-right endpoint)
    const handleRightBot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    handleRightBot.setAttribute('cx', crb);
    handleRightBot.setAttribute('cy', h - 8);
    handleRightBot.setAttribute('r', '4');
    handleRightBot.setAttribute('fill', '#ff9100');
    handleRightBot.setAttribute('stroke', '#ffffff');
    handleRightBot.setAttribute('stroke-width', '1.2');
    handleRightBot.setAttribute('class', 'lateral-corner-handle');
    handleRightBot.id = 'svg-handle-right-bot';
    this.svgOverlay.appendChild(handleRightBot);

    const hitRightBot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    hitRightBot.setAttribute('cx', crb);
    hitRightBot.setAttribute('cy', h - 8);
    hitRightBot.setAttribute('r', '14');
    hitRightBot.setAttribute('fill', 'transparent');
    hitRightBot.style.cursor = 'ew-resize';
    hitRightBot.id = 'svg-hit-right-bot';
    this.bindLateralDrag(hitRightBot, item, 'right_bot');
    this.svgOverlay.appendChild(hitRightBot);
  }

  updateLateralSVG(item) {
    const w = item.width;
    const h = item.height;
    const clt = item.crop_left_top !== undefined ? item.crop_left_top : (item.crop_left || 0);
    const clb = item.crop_left_bot !== undefined ? item.crop_left_bot : (item.crop_left || 0);
    const crt = item.crop_right_top !== undefined ? item.crop_right_top : (item.crop_right || w);
    const crb = item.crop_right_bot !== undefined ? item.crop_right_bot : (item.crop_right || w);

    const curtainL = document.getElementById('svg-curtain-left');
    if (curtainL) curtainL.setAttribute('points', `0,0 ${clt},0 ${clb},${h} 0,${h}`);

    const lineL = document.getElementById('svg-line-left');
    if (lineL) { lineL.setAttribute('x1', clt); lineL.setAttribute('x2', clb); }

    const hitL = document.getElementById('svg-hit-left');
    if (hitL) { hitL.setAttribute('x1', clt); hitL.setAttribute('x2', clb); }

    const handleLT = document.getElementById('svg-handle-left-top');
    if (handleLT) handleLT.setAttribute('cx', clt);
    const hitLT = document.getElementById('svg-hit-left-top');
    if (hitLT) hitLT.setAttribute('cx', clt);

    const handleLB = document.getElementById('svg-handle-left-bot');
    if (handleLB) handleLB.setAttribute('cx', clb);
    const hitLB = document.getElementById('svg-hit-left-bot');
    if (hitLB) hitLB.setAttribute('cx', clb);

    const curtainR = document.getElementById('svg-curtain-right');
    if (curtainR) curtainR.setAttribute('points', `${crt},0 ${w},0 ${w},${h} ${crb},${h}`);

    const lineR = document.getElementById('svg-line-right');
    if (lineR) { lineR.setAttribute('x1', crt); lineR.setAttribute('x2', crb); }

    const hitR = document.getElementById('svg-hit-right');
    if (hitR) { hitR.setAttribute('x1', crt); hitR.setAttribute('x2', crb); }

    const handleRT = document.getElementById('svg-handle-right-top');
    if (handleRT) handleRT.setAttribute('cx', crt);
    const hitRT = document.getElementById('svg-hit-right-top');
    if (hitRT) hitRT.setAttribute('cx', crt);

    const handleRB = document.getElementById('svg-handle-right-bot');
    if (handleRB) handleRB.setAttribute('cx', crb);
    const hitRB = document.getElementById('svg-hit-right-bot');
    if (hitRB) hitRB.setAttribute('cx', crb);
  }

  isSelected(curveType, idx) {
    return this.selectedNodes.some(n => n.curveType === curveType && n.idx === idx);
  }

  selectNodesInBox(minX, minY, maxX, maxY, item) {
    this.selectedNodes = [];
    const topPts = item.y_top_points || [];
    const botPts = item.y_bot_points || [];

    topPts.forEach((pt, idx) => {
      if (pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
        this.selectedNodes.push({ curveType: 'top', idx });
      }
    });

    botPts.forEach((pt, idx) => {
      if (pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
        this.selectedNodes.push({ curveType: 'bot', idx });
      }
    });

    this.updateSelectedNodesVisual();
  }

  updateSelectedNodesVisual() {
    this.svgOverlay.querySelectorAll('#svg-handles-group circle.vector-handle').forEach(circle => {
      const curve = circle.getAttribute('data-curve');
      const idx = parseInt(circle.getAttribute('data-idx'), 10);
      if (this.isSelected(curve, idx)) {
        circle.classList.add('node-selected');
        circle.setAttribute('r', '4.0');
        circle.setAttribute('stroke', '#ffd700');
        circle.setAttribute('stroke-width', '1.5');
      } else {
        circle.classList.remove('node-selected');
        circle.setAttribute('r', '2.75');
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '1.0');
      }
    });

    const currentItem = this.queue ? this.queue[this.currentIndex] : null;
    if (currentItem) {
      this.renderSelectionRotationWidget(currentItem);
    }
  }

  renderSelectionRotationWidget(item) {
    if (!this.svgOverlay || !item) return;

    let rotWidget = document.getElementById('svg-selection-rotate-widget');

    if (!this.selectedNodes || this.selectedNodes.length < 2) {
      if (rotWidget) rotWidget.remove();
      return;
    }

    const selectedPts = this.selectedNodes.map(node => {
      const pt = (node.curveType === 'top') ? item.y_top_points[node.idx] : item.y_bot_points[node.idx];
      return { curveType: node.curveType, idx: node.idx, x: pt.x, y: pt.y };
    }).filter(p => p.x !== undefined && p.y !== undefined);

    if (selectedPts.length < 2) {
      if (rotWidget) rotWidget.remove();
      return;
    }

    const hasTop = selectedPts.some(p => p.curveType === 'top');
    const hasBot = selectedPts.some(p => p.curveType === 'bot');
    const isBottomCurve = hasBot && !hasTop;

    const pivotX = selectedPts.reduce((sum, p) => sum + p.x, 0) / selectedPts.length;

    const handleX = pivotX;
    let handleY;
    let yRot, ySlider;

    if (isBottomCurve) {
      // Below bottom curve: Slider near curve (Row 1), Rotate further down (Row 2)
      const maxY = Math.max(...selectedPts.map(p => p.y));
      handleY = Math.min(item.height - 24, maxY + 46);
      ySlider = -12;
      yRot = +16;
    } else {
      // Above top curve: Rotate further up (Row 1), Slider near curve (Row 2)
      const minY = Math.min(...selectedPts.map(p => p.y));
      handleY = Math.max(24, minY - 46);
      yRot = -16;
      ySlider = +12;
    }

    if (!rotWidget) {
      rotWidget = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      rotWidget.id = 'svg-selection-rotate-widget';

      // 1. Detached Element 1: Circular Rotation Badge (Row 1)
      const rotG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      rotG.id = 'svg-rot-subwidget';

      const rotHit = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      rotHit.setAttribute('cx', '0');
      rotHit.setAttribute('cy', '0');
      rotHit.setAttribute('r', '20');
      rotHit.setAttribute('fill', 'transparent');
      rotHit.setAttribute('class', 'rotate-hit-target');
      rotG.appendChild(rotHit);

      const rotBadge = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      rotBadge.setAttribute('cx', '0');
      rotBadge.setAttribute('cy', '0');
      rotBadge.setAttribute('r', '12');
      rotBadge.setAttribute('fill', '#090d16');
      rotBadge.setAttribute('stroke', '#ffd700');
      rotBadge.setAttribute('stroke-width', '1.8');
      rotBadge.setAttribute('class', 'rotate-visual-badge');
      rotBadge.setAttribute('style', 'filter: drop-shadow(0 0 8px rgba(255, 215, 0, 0.9)); pointer-events: none;');
      rotG.appendChild(rotBadge);

      // Dual rotation arrows centered at (0, 0)
      const arc1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      arc1.setAttribute('d', 'M -6 -1 A 6 6 0 0 1 4.5 -4');
      arc1.setAttribute('fill', 'none');
      arc1.setAttribute('stroke', '#ffd700');
      arc1.setAttribute('stroke-width', '1.5');
      arc1.setAttribute('stroke-linecap', 'round');
      arc1.setAttribute('style', 'pointer-events: none;');
      rotG.appendChild(arc1);

      const head1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      head1.setAttribute('d', 'M 1.5 -5.8 L 5.5 -3.8 L 3.5 -1.2 Z');
      head1.setAttribute('fill', '#ffd700');
      head1.setAttribute('stroke', '#ffd700');
      head1.setAttribute('stroke-width', '0.5');
      head1.setAttribute('style', 'pointer-events: none;');
      rotG.appendChild(head1);

      const arc2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      arc2.setAttribute('d', 'M 6 1 A 6 6 0 0 1 -4.5 4');
      arc2.setAttribute('fill', 'none');
      arc2.setAttribute('stroke', '#ffd700');
      arc2.setAttribute('stroke-width', '1.5');
      arc2.setAttribute('stroke-linecap', 'round');
      arc2.setAttribute('style', 'pointer-events: none;');
      rotG.appendChild(arc2);

      const head2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      head2.setAttribute('d', 'M -1.5 5.8 L -5.5 3.8 L -3.5 1.2 Z');
      head2.setAttribute('fill', '#ffd700');
      head2.setAttribute('stroke', '#ffd700');
      head2.setAttribute('stroke-width', '0.5');
      head2.setAttribute('style', 'pointer-events: none;');
      rotG.appendChild(head2);

      const onRotateStart = (e) => {
        e.stopPropagation();
        e.preventDefault();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
        const startSvg = this.clientToSvg(clientX, clientY);

        const initialPoints = this.selectedNodes.map(node => {
          const pt = (node.curveType === 'top') ? item.y_top_points[node.idx] : item.y_bot_points[node.idx];
          return { curveType: node.curveType, idx: node.idx, x: pt.x, y: pt.y };
        });

        const curPivotX = initialPoints.reduce((sum, p) => sum + p.x, 0) / initialPoints.length;
        const curPivotY = initialPoints.reduce((sum, p) => sum + p.y, 0) / initialPoints.length;
        const startAngle = Math.atan2(startSvg.y - curPivotY, startSvg.x - curPivotX);

        this.activeDrag = {
          type: 'multi_node_rotate',
          item,
          pivotX: curPivotX,
          pivotY: curPivotY,
          startAngle,
          initialPoints
        };

        if (this.svgOverlay) {
          this.svgOverlay.classList.add('is-rotating-nodes');
        }
      };

      rotHit.onmousedown = onRotateStart;
      rotHit.ontouchstart = onRotateStart;
      rotWidget.appendChild(rotG);

      // 2. Detached Element 2: Rounding / Curvature Slider Pill (Row 2)
      const roundG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      roundG.id = 'svg-round-slider-subwidget';

      // Slider Pill Background
      const sliderPill = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      sliderPill.setAttribute('x', '-40');
      sliderPill.setAttribute('y', '-9');
      sliderPill.setAttribute('width', '80');
      sliderPill.setAttribute('height', '18');
      sliderPill.setAttribute('rx', '9');
      sliderPill.setAttribute('ry', '9');
      sliderPill.setAttribute('fill', 'rgba(10, 15, 29, 0.94)');
      sliderPill.setAttribute('stroke', '#00f2fe');
      sliderPill.setAttribute('stroke-width', '1.2');
      sliderPill.setAttribute('style', 'filter: drop-shadow(0 3px 10px rgba(0,0,0,0.5)); pointer-events: none;');
      roundG.appendChild(sliderPill);

      // Curved indicator left (bend down)
      const iconL = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      iconL.setAttribute('d', 'M -32 2.5 Q -29 -2.5 -26 2.5');
      iconL.setAttribute('fill', 'none');
      iconL.setAttribute('stroke', '#00f2fe');
      iconL.setAttribute('stroke-width', '1.2');
      iconL.setAttribute('stroke-linecap', 'round');
      iconL.setAttribute('opacity', '0.75');
      iconL.setAttribute('style', 'pointer-events: none;');
      roundG.appendChild(iconL);

      // Slider Track Rail (40px width: -20 to +20)
      const track = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      track.setAttribute('x', '-20');
      track.setAttribute('y', '-2.5');
      track.setAttribute('width', '40');
      track.setAttribute('height', '5');
      track.setAttribute('rx', '2.5');
      track.setAttribute('fill', '#050811');
      track.setAttribute('stroke', '#334155');
      track.setAttribute('stroke-width', '0.8');
      track.setAttribute('style', 'pointer-events: none;');
      roundG.appendChild(track);

      // Center notch
      const notch = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      notch.setAttribute('x1', '0');
      notch.setAttribute('y1', '-3');
      notch.setAttribute('x2', '0');
      notch.setAttribute('y2', '3');
      notch.setAttribute('stroke', '#ffffff');
      notch.setAttribute('stroke-width', '1');
      notch.setAttribute('opacity', '0.45');
      notch.setAttribute('style', 'pointer-events: none;');
      roundG.appendChild(notch);

      // Curved indicator right (bend up)
      const iconR = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      iconR.setAttribute('d', 'M 26 -2.5 Q 29 2.5 32 -2.5');
      iconR.setAttribute('fill', 'none');
      iconR.setAttribute('stroke', '#00f2fe');
      iconR.setAttribute('stroke-width', '1.2');
      iconR.setAttribute('stroke-linecap', 'round');
      iconR.setAttribute('opacity', '0.75');
      iconR.setAttribute('style', 'pointer-events: none;');
      roundG.appendChild(iconR);

      // Slider Hit Target
      const roundHit = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      roundHit.setAttribute('x', '-40');
      roundHit.setAttribute('y', '-11');
      roundHit.setAttribute('width', '80');
      roundHit.setAttribute('height', '22');
      roundHit.setAttribute('fill', 'transparent');
      roundHit.setAttribute('class', 'round-hit-target');
      roundHit.setAttribute('style', 'cursor: ew-resize;');
      roundG.appendChild(roundHit);

      // Draggable Thumb Knob
      const thumbG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      thumbG.id = 'svg-round-thumb-g';
      thumbG.setAttribute('transform', 'translate(0, 0)');

      const thumbCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      thumbCircle.setAttribute('r', '5.5');
      thumbCircle.setAttribute('fill', '#00f2fe');
      thumbCircle.setAttribute('stroke', '#ffffff');
      thumbCircle.setAttribute('stroke-width', '1.4');
      thumbCircle.setAttribute('class', 'round-thumb-visual');
      thumbCircle.setAttribute('style', 'filter: drop-shadow(0 0 6px rgba(0, 242, 254, 0.9)); pointer-events: none;');
      thumbG.appendChild(thumbCircle);

      const thumbDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      thumbDot.setAttribute('r', '1.3');
      thumbDot.setAttribute('fill', '#0f172a');
      thumbDot.setAttribute('style', 'pointer-events: none;');
      thumbG.appendChild(thumbDot);

      roundG.appendChild(thumbG);

      const onRoundStart = (e) => {
        e.stopPropagation();
        e.preventDefault();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
        const startSvg = this.clientToSvg(clientX, 0);

        const initialPoints = this.selectedNodes.map(node => {
          const pt = (node.curveType === 'top') ? item.y_top_points[node.idx] : item.y_bot_points[node.idx];
          return { curveType: node.curveType, idx: node.idx, x: pt.x, y: pt.y };
        });

        const minX = Math.min(...initialPoints.map(p => p.x));
        const maxX = Math.max(...initialPoints.map(p => p.x));
        const spanX = Math.max(1, maxX - minX);

        this.activeDrag = {
          type: 'multi_node_round',
          item,
          startSvgX: startSvg.x,
          initialPoints,
          minX,
          spanX
        };

        if (this.svgOverlay) {
          this.svgOverlay.classList.add('is-rotating-nodes');
        }
      };

      roundHit.onmousedown = onRoundStart;
      roundHit.ontouchstart = onRoundStart;
      rotWidget.appendChild(roundG);

      this.svgOverlay.appendChild(rotWidget);
    }

    const rotG = document.getElementById('svg-rot-subwidget');
    if (rotG) rotG.setAttribute('transform', `translate(0, ${yRot})`);

    const roundG = document.getElementById('svg-round-slider-subwidget');
    if (roundG) roundG.setAttribute('transform', `translate(0, ${ySlider})`);

    rotWidget.setAttribute('transform', `translate(${handleX}, ${handleY})`);
  }

  clearSelection() {
    this.selectedNodes = [];
    if (this.svgOverlay) {
      this.updateSelectedNodesVisual();
      const selBox = document.getElementById('svg-selection-box');
      if (selBox) selBox.remove();
      const rotWidget = document.getElementById('svg-selection-rotate-widget');
      if (rotWidget) rotWidget.remove();
    }
  }

  insertNodeOnCurve(item, curveType, clickX, clickY) {
    const pts = (curveType === 'top') ? item.y_top_points : item.y_bot_points;
    if (!pts || pts.length === 0) return;

    const cx = Math.max(1, Math.min(item.width - 2, clickX));
    const cy = Math.max(0, Math.min(item.height - 1, clickY));

    // Avoid inserting duplicate node on top of an existing node within 4px
    if (pts.some(p => Math.abs(p.x - cx) < 4)) {
      return;
    }

    pts.push({ x: cx, y: cy });
    pts.sort((a, b) => a.x - b.x);

    const newIdx = pts.findIndex(p => p.x === cx && p.y === cy);
    this.selectedNodes = [{ curveType, idx: newIdx }];

    this.renderVectorsAndCrop(item);
  }

  removeNodeFromCurve(item, curveType, pointIdx) {
    const pts = (curveType === 'top') ? item.y_top_points : item.y_bot_points;
    if (!pts || pts.length <= 4) {
      return;
    }
    // Prevent deleting left or right boundary edge anchors
    if (pointIdx === 0 || pointIdx === pts.length - 1) {
      return;
    }

    pts.splice(pointIdx, 1);
    this.clearSelection();
    this.renderVectorsAndCrop(item);
  }

  deleteSelectedNodes() {
    const item = this.queue[this.currentIndex];
    if (!item || !this.selectedNodes || this.selectedNodes.length === 0) return;

    let modified = false;
    const topIndices = this.selectedNodes.filter(n => n.curveType === 'top').map(n => n.idx).sort((a, b) => b - a);
    const botIndices = this.selectedNodes.filter(n => n.curveType === 'bot').map(n => n.idx).sort((a, b) => b - a);

    topIndices.forEach(idx => {
      if (item.y_top_points.length > 4 && idx > 0 && idx < item.y_top_points.length - 1) {
        item.y_top_points.splice(idx, 1);
        modified = true;
      }
    });

    botIndices.forEach(idx => {
      if (item.y_bot_points.length > 4 && idx > 0 && idx < item.y_bot_points.length - 1) {
        item.y_bot_points.splice(idx, 1);
        modified = true;
      }
    });

    if (modified) {
      this.clearSelection();
      this.renderVectorsAndCrop(item);
    }
  }

  handleGlobalDragMove(e) {
    if (!this.activeDrag) return;
    if (e.cancelable) e.preventDefault();

    if (this.svgOverlay) this.svgOverlay.classList.add('is-dragging-nodes');

    const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
    const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
    const curSvg = this.clientToSvg(clientX, clientY);

    const { type, item, startSvgX, startSvgY, startClientX, startClientY, initialTopYs, initialBotYs, initialCropLeft, initialCropRight, pointIdx, curveType, initialYs } = this.activeDrag;

    if (type === 'marquee_select') {
      const deltaPx = Math.hypot(clientX - startClientX, clientY - startClientY);
      if (deltaPx > 3) {
        this.activeDrag.moved = true;
        const minX = Math.max(0, Math.min(startSvgX, curSvg.x));
        const maxX = Math.min(item.width, Math.max(startSvgX, curSvg.x));
        const minY = Math.max(0, Math.min(startSvgY, curSvg.y));
        const maxY = Math.min(item.height, Math.max(startSvgY, curSvg.y));

        let box = document.getElementById('svg-selection-box');
        if (!box) {
          box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          box.id = 'svg-selection-box';
          box.setAttribute('fill', 'rgba(0, 242, 254, 0.16)');
          box.setAttribute('stroke', '#00f2fe');
          box.setAttribute('stroke-width', '1.5');
          box.setAttribute('stroke-dasharray', '4 3');
          box.style.pointerEvents = 'none';
          this.svgOverlay.appendChild(box);
        }
        box.setAttribute('x', minX);
        box.setAttribute('y', minY);
        box.setAttribute('width', Math.max(0.1, maxX - minX));
        box.setAttribute('height', Math.max(0.1, maxY - minY));

        this.selectNodesInBox(minX, minY, maxX, maxY, item);
      }
    } else if (type === 'multi_node_drag') {
      const deltaSvgY = curSvg.y - startSvgY;
      this.selectedNodes.forEach((node, i) => {
        const clampedY = Math.max(0, Math.min(item.height - 1, initialYs[i] + deltaSvgY));
        if (node.curveType === 'top') {
          item.y_top_points[node.idx].y = clampedY;
        } else {
          item.y_bot_points[node.idx].y = clampedY;
        }
        this.svgOverlay.querySelectorAll(`#svg-handles-group circle[data-curve="${node.curveType}"][data-idx="${node.idx}"]`).forEach(c => {
          c.setAttribute('cy', clampedY);
        });
      });
      this.updateSVGPaths(item);
      this.renderSelectionRotationWidget(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'multi_node_rotate') {
      const { pivotX, pivotY, startAngle, initialPoints } = this.activeDrag;
      const currentAngle = Math.atan2(curSvg.y - pivotY, curSvg.x - pivotX);
      const deltaAngle = currentAngle - startAngle;

      initialPoints.forEach(p => {
        const dx = p.x - pivotX;
        const dy = p.y - pivotY;
        const rotatedDy = dx * Math.sin(deltaAngle) + dy * Math.cos(deltaAngle);
        const newY = Math.max(0, Math.min(item.height - 1, pivotY + rotatedDy));

        if (p.curveType === 'top') {
          item.y_top_points[p.idx].y = newY;
        } else {
          item.y_bot_points[p.idx].y = newY;
        }

        this.svgOverlay.querySelectorAll(`#svg-handles-group circle[data-curve="${p.curveType}"][data-idx="${p.idx}"]`).forEach(c => {
          c.setAttribute('cy', newY);
        });
      });

      this.updateSVGPaths(item);
      this.renderSelectionRotationWidget(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'multi_node_round') {
      const { startSvgX, initialPoints, minX, spanX } = this.activeDrag;
      const deltaSvgX = curSvg.x - startSvgX;
      // Slider rail is 40px wide (-20 to +20)
      const clampedDeltaX = Math.max(-20, Math.min(20, deltaSvgX));

      const thumbEl = document.getElementById('svg-round-thumb-g');
      if (thumbEl) {
        thumbEl.setAttribute('transform', `translate(${clampedDeltaX}, 0)`);
      }

      const factor = clampedDeltaX / 20; // range: -1.0 to +1.0
      const maxArch = 35; // Maximum pixel displacement at the apex of the curve

      initialPoints.forEach(p => {
        const t = (p.x - minX) / spanX; // 0.0 to 1.0
        const archWeight = Math.sin(Math.PI * t);
        // Dragging left (negative deltaX): curves down (increases Y in SVG)
        // Dragging right (positive deltaX): curves up (decreases Y in SVG)
        const newY = Math.max(0, Math.min(item.height - 1, p.y - factor * maxArch * archWeight));

        if (p.curveType === 'top') {
          item.y_top_points[p.idx].y = newY;
        } else {
          item.y_bot_points[p.idx].y = newY;
        }

        this.svgOverlay.querySelectorAll(`#svg-handles-group circle[data-curve="${p.curveType}"][data-idx="${p.idx}"]`).forEach(c => {
          c.setAttribute('cy', newY);
        });
      });

      this.updateSVGPaths(item);
      this.renderSelectionRotationWidget(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'single_node_proportional') {
      const deltaSvgY = curSvg.y - startSvgY;
      const pts = curveType === 'top' ? item.y_top_points : item.y_bot_points;
      const sigma = 8.0; // Surgical localized Gaussian influence radius (tight, responsive point editing)

      pts.forEach((pt, i) => {
        if (this.activeDrag.isStrict) {
          if (i === pointIdx) {
            pt.y = Math.max(0, Math.min(item.height - 1, this.activeDrag.initialCurveYs[i] + deltaSvgY));
          }
        } else {
          const dx = pt.x - this.activeDrag.targetPointX;
          const weight = Math.exp(-(dx * dx) / (2 * sigma * sigma));
          pt.y = Math.max(0, Math.min(item.height - 1, this.activeDrag.initialCurveYs[i] + deltaSvgY * weight));
        }
      });

      this.svgOverlay.querySelectorAll(`#svg-handles-group circle[data-curve="${curveType}"]`).forEach(c => {
        const idx = parseInt(c.getAttribute('data-idx'), 10);
        if (pts[idx]) c.setAttribute('cy', pts[idx].y);
      });

      this.updateSVGPaths(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'point') {
      const clampedY = Math.max(0, Math.min(item.height - 1, curSvg.y));
      if (curveType === 'top') {
        item.y_top_points[pointIdx].y = clampedY;
      } else {
        item.y_bot_points[pointIdx].y = clampedY;
      }
      this.svgOverlay.querySelectorAll(`#svg-handles-group circle[data-curve="${curveType}"][data-idx="${pointIdx}"]`).forEach(c => {
        c.setAttribute('cy', clampedY);
      });
      this.updateSVGPaths(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'curve_top' || type === 'curve_bot' || type === 'curve_both') {
      const deltaSvgY = curSvg.y - startSvgY;
      if (type === 'curve_top' || type === 'curve_both') {
        item.y_top_points.forEach((pt, i) => {
          pt.y = Math.max(0, Math.min(item.height - 1, initialTopYs[i] + deltaSvgY));
        });
      }
      if (type === 'curve_bot' || type === 'curve_both') {
        item.y_bot_points.forEach((pt, i) => {
          pt.y = Math.max(0, Math.min(item.height - 1, initialBotYs[i] + deltaSvgY));
        });
      }
      // Update both hit targets and visual circles
      this.svgOverlay.querySelectorAll('#svg-handles-group circle').forEach(circle => {
        const curve = circle.getAttribute('data-curve');
        const idx = parseInt(circle.getAttribute('data-idx'), 10);
        if (curve === 'top' && item.y_top_points[idx]) {
          circle.setAttribute('cy', item.y_top_points[idx].y);
        } else if (curve === 'bot' && item.y_bot_points[idx]) {
          circle.setAttribute('cy', item.y_bot_points[idx].y);
        }
      });
      this.updateSVGPaths(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_left_line') {
      const deltaX = curSvg.x - startSvgX;
      const maxAllowedTop = Math.min(item.crop_right_top - 20, item.width - 20);
      const maxAllowedBot = Math.min(item.crop_right_bot - 20, item.width - 20);
      item.crop_left_top = Math.max(0, Math.min(maxAllowedTop, Math.round(this.activeDrag.initialCropLeftTop + deltaX)));
      item.crop_left_bot = Math.max(0, Math.min(maxAllowedBot, Math.round(this.activeDrag.initialCropLeftBot + deltaX)));
      item.crop_left = Math.round((item.crop_left_top + item.crop_left_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_left_top') {
      const deltaX = curSvg.x - startSvgX;
      const maxAllowed = Math.min(item.crop_right_top - 20, item.width - 20);
      item.crop_left_top = Math.max(0, Math.min(maxAllowed, Math.round(this.activeDrag.initialCropLeftTop + deltaX)));
      item.crop_left = Math.round((item.crop_left_top + item.crop_left_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_left_bot') {
      const deltaX = curSvg.x - startSvgX;
      const maxAllowed = Math.min(item.crop_right_bot - 20, item.width - 20);
      item.crop_left_bot = Math.max(0, Math.min(maxAllowed, Math.round(this.activeDrag.initialCropLeftBot + deltaX)));
      item.crop_left = Math.round((item.crop_left_top + item.crop_left_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_right_line') {
      const deltaX = curSvg.x - startSvgX;
      const minAllowedTop = Math.max(item.crop_left_top + 20, 20);
      const minAllowedBot = Math.max(item.crop_left_bot + 20, 20);
      item.crop_right_top = Math.min(item.width, Math.max(minAllowedTop, Math.round(this.activeDrag.initialCropRightTop + deltaX)));
      item.crop_right_bot = Math.min(item.width, Math.max(minAllowedBot, Math.round(this.activeDrag.initialCropRightBot + deltaX)));
      item.crop_right = Math.round((item.crop_right_top + item.crop_right_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_right_top') {
      const deltaX = curSvg.x - startSvgX;
      const minAllowed = Math.max(item.crop_left_top + 20, 20);
      item.crop_right_top = Math.min(item.width, Math.max(minAllowed, Math.round(this.activeDrag.initialCropRightTop + deltaX)));
      item.crop_right = Math.round((item.crop_right_top + item.crop_right_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    } else if (type === 'lateral_right_bot') {
      const deltaX = curSvg.x - startSvgX;
      const minAllowed = Math.max(item.crop_left_bot + 20, 20);
      item.crop_right_bot = Math.min(item.width, Math.max(minAllowed, Math.round(this.activeDrag.initialCropRightBot + deltaX)));
      item.crop_right = Math.round((item.crop_right_top + item.crop_right_bot) / 2);
      this.updateLateralSVG(item);
      this.renderLiveCropCanvas(item);
    }
  }

  handleGlobalDragEnd(e) {
    if (this.svgOverlay) this.svgOverlay.classList.remove('is-dragging-nodes', 'is-rotating-nodes');
    if (this.activeDrag) {
      if (this.activeDrag.type === 'marquee_select') {
        const box = document.getElementById('svg-selection-box');
        if (box) box.remove();
        if (!this.activeDrag.moved) {
          // Clicked in empty space -> Let go / Deselect all
          this.clearSelection();
        }
      }
      if (this.activeDrag.type === 'multi_node_round') {
        const thumbEl = document.getElementById('svg-round-thumb-g');
        if (thumbEl) {
          thumbEl.setAttribute('transform', 'translate(0, 0)');
        }
      }
      const item = this.activeDrag.item || (this.queue ? this.queue[this.currentIndex] : null);
      this.activeDrag = null;
      if (item && this.selectedNodes && this.selectedNodes.length >= 2) {
        this.renderSelectionRotationWidget(item);
      }
      if (e && e.clientX !== undefined && e.clientY !== undefined && item) {
        this.updateProximityNodeHover(e.clientX, e.clientY, item);
      }
    }
  }

  updateProximityNodeHover(clientX, clientY, item) {
    if (!this.svgOverlay || !item || this.activeDrag) return;
    const topPts = item.y_top_points || [];
    const botPts = item.y_bot_points || [];
    if (topPts.length === 0 && botPts.length === 0) return;

    const svgPt = this.clientToSvg(clientX, clientY);
    const sx = svgPt.x;
    const sy = svgPt.y;

    let minTopDist = Infinity;
    let closestTopIdx = -1;
    for (let i = 0; i < topPts.length; i++) {
      const d = Math.hypot(sx - topPts[i].x, sy - topPts[i].y);
      if (d < minTopDist) {
        minTopDist = d;
        closestTopIdx = i;
      }
    }

    let minBotDist = Infinity;
    let closestBotIdx = -1;
    for (let i = 0; i < botPts.length; i++) {
      const d = Math.hypot(sx - botPts[i].x, sy - botPts[i].y);
      if (d < minBotDist) {
        minBotDist = d;
        closestBotIdx = i;
      }
    }

    const PROXIMITY_THRESHOLD = 90; // Generous activation range in SVG pixels
    let activeCurve = null;
    let activeIdx = -1;
    let bestDist = Infinity;

    if (minTopDist <= PROXIMITY_THRESHOLD && minTopDist < bestDist) {
      bestDist = minTopDist;
      activeCurve = 'top';
      activeIdx = closestTopIdx;
    }
    if (minBotDist <= PROXIMITY_THRESHOLD && minBotDist < bestDist) {
      bestDist = minBotDist;
      activeCurve = 'bot';
      activeIdx = closestBotIdx;
    }

    const circles = this.svgOverlay.querySelectorAll('#svg-handles-group circle.vector-handle');
    circles.forEach(circle => {
      const curve = circle.getAttribute('data-curve');
      const idx = parseInt(circle.getAttribute('data-idx'), 10);

      circle.classList.remove('handle-hover-primary', 'handle-hover-adjacent');

      if (activeCurve && curve === activeCurve) {
        const diff = Math.abs(idx - activeIdx);
        if (diff === 0) {
          // Exact closest node to cursor
          circle.classList.add('handle-hover-primary');
        } else if (diff <= 2) {
          // Adjacent 2 nodes on the line (left and right)
          circle.classList.add('handle-hover-adjacent');
        }
      }
    });
  }

  clearProximityNodeHover() {
    if (!this.svgOverlay) return;
    const circles = this.svgOverlay.querySelectorAll('#svg-handles-group circle.vector-handle');
    circles.forEach(circle => {
      circle.classList.remove('handle-hover-primary', 'handle-hover-adjacent');
    });
  }

  bindCurveGlobalDrag(el, item, targetMode) {
    const onStart = (e) => {
      e.stopPropagation();
      e.preventDefault();
      const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
      const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
      const startSvg = this.clientToSvg(clientX, clientY);
      this.activeDrag = {
        type: targetMode === 'both' ? 'curve_both' : (targetMode === 'top' ? 'curve_top' : 'curve_bot'),
        item,
        startSvgY: startSvg.y,
        initialTopYs: (item.y_top_points || []).map(p => p.y),
        initialBotYs: (item.y_bot_points || []).map(p => p.y)
      };
    };

    el.addEventListener('mousedown', onStart);
    el.addEventListener('touchstart', onStart, { passive: false });
  }

  bindLateralDrag(el, item, dragMode) {
    const onStart = (e) => {
      e.stopPropagation();
      e.preventDefault();
      const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
      const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
      const startSvg = this.clientToSvg(clientX, clientY);
      this.activeDrag = {
        type: 'lateral_' + dragMode,
        item,
        startSvgX: startSvg.x,
        initialCropLeftTop: item.crop_left_top !== undefined ? item.crop_left_top : (item.crop_left || 0),
        initialCropLeftBot: item.crop_left_bot !== undefined ? item.crop_left_bot : (item.crop_left || 0),
        initialCropRightTop: item.crop_right_top !== undefined ? item.crop_right_top : (item.crop_right || item.width),
        initialCropRightBot: item.crop_right_bot !== undefined ? item.crop_right_bot : (item.crop_right || item.width)
      };
    };

    el.addEventListener('mousedown', onStart);
    el.addEventListener('touchstart', onStart, { passive: false });
  }

  bindPointDrag(hitTarget, item, curveType, pointIdx) {
    const onStart = (e) => {
      e.stopPropagation();
      e.preventDefault();
      const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
      const clientY = e.clientY || (e.touches && e.touches[0].clientY) || 0;
      const startSvg = this.clientToSvg(clientX, clientY);
      const pts = curveType === 'top' ? item.y_top_points : item.y_bot_points;

      if (this.isSelected(curveType, pointIdx)) {
        // Drag all currently selected nodes together
        const initialYs = this.selectedNodes.map(n => (n.curveType === 'top' ? item.y_top_points[n.idx].y : item.y_bot_points[n.idx].y));
        this.activeDrag = {
          type: 'multi_node_drag',
          item,
          startSvgY: startSvg.y,
          initialYs
        };
      } else {
        if (!e.shiftKey) {
          this.selectedNodes = [{ curveType, idx: pointIdx }];
          this.updateSelectedNodesVisual();
        } else {
          this.selectedNodes.push({ curveType, idx: pointIdx });
          this.updateSelectedNodesVisual();
        }
        this.activeDrag = {
          type: 'single_node_proportional',
          item,
          curveType,
          pointIdx,
          targetPointX: pts[pointIdx].x,
          startSvgY: startSvg.y,
          initialCurveYs: pts.map(p => p.y),
          isStrict: e.altKey || e.ctrlKey
        };
      }
    };

    hitTarget.addEventListener('mousedown', onStart);
    hitTarget.addEventListener('touchstart', onStart, { passive: false });
  }

  updateSVGPaths(item) {
    const topPts = item.y_top_points || [];
    const botPts = item.y_bot_points || [];
    const topPath = document.getElementById('svg-top-path');
    const topHit = document.getElementById('svg-top-hit');
    const botPath = document.getElementById('svg-bot-path');
    const botHit = document.getElementById('svg-bot-hit');
    const polyEl = document.getElementById('svg-retinal-band');
    const topGrabG = document.getElementById('svg-top-grab-handle');
    const botGrabG = document.getElementById('svg-bot-grab-handle');

    if (topPts.length > 0) {
      const topD = this.pointsToSmoothPathD(topPts);
      if (topPath) topPath.setAttribute('d', topD);
      if (topHit) topHit.setAttribute('d', topD);
      if (topGrabG) {
        const topPeakY = this.getCentralPeakY(topPts, item.width, 'top');
        topGrabG.setAttribute('transform', `translate(${item.width / 2}, ${Math.max(12, topPeakY - 48)})`);
      }
    }
    if (botPts.length > 0) {
      const botD = this.pointsToSmoothPathD(botPts);
      if (botPath) botPath.setAttribute('d', botD);
      if (botHit) botHit.setAttribute('d', botD);
      if (botGrabG) {
        const botPeakY = this.getCentralPeakY(botPts, item.width, 'bot');
        botGrabG.setAttribute('transform', `translate(${item.width / 2}, ${Math.min(item.height - 12, botPeakY + 48)})`);
      }
    }
    if (polyEl && topPts.length > 0 && botPts.length > 0) {
      const polyD = this.pointsToSmoothBandPolygonD(topPts, botPts);
      polyEl.setAttribute('d', polyD);
    }
  }

  renderLiveCropCanvas(item) {
    if (!this.cropPreviewCanvas || !this.imgRaw || !item) return;
    const w = item.width;
    const h = item.height;
    const topPts = item.y_top_points || [];
    const botPts = item.y_bot_points || [];
    if (topPts.length === 0 || botPts.length === 0) return;

    const clt = item.crop_left_top !== undefined ? item.crop_left_top : (item.crop_left || 0);
    const clb = item.crop_left_bot !== undefined ? item.crop_left_bot : (item.crop_left || 0);
    const crt = item.crop_right_top !== undefined ? item.crop_right_top : (item.crop_right || w);
    const crb = item.crop_right_bot !== undefined ? item.crop_right_bot : (item.crop_right || w);

    // 1. Render Left Overlaid Canvas (Mode dependent)
    const canvas = this.cropPreviewCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    this.traceSmoothPathToCanvas(ctx, topPts, botPts);

    if (this.viewMode === 'mask') {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      // Zero Left Slanted Polygon
      if (clt > 0 || clb > 0) {
        ctx.beginPath();
        ctx.moveTo(0, 0); ctx.lineTo(clt, 0); ctx.lineTo(clb, h); ctx.lineTo(0, h);
        ctx.closePath(); ctx.fillStyle = '#000000'; ctx.fill();
      }
      // Zero Right Slanted Polygon
      if (crt < w || crb < w) {
        ctx.beginPath();
        ctx.moveTo(crt, 0); ctx.lineTo(w, 0); ctx.lineTo(w, h); ctx.lineTo(crb, h);
        ctx.closePath(); ctx.fillStyle = '#000000'; ctx.fill();
      }
    } else {
      ctx.clip();
      ctx.drawImage(this.imgRaw, 0, 0, w, h);
      // Zero Left Slanted Polygon
      if (clt > 0 || clb > 0) {
        ctx.beginPath();
        ctx.moveTo(0, 0); ctx.lineTo(clt, 0); ctx.lineTo(clb, h); ctx.lineTo(0, h);
        ctx.closePath(); ctx.fillStyle = '#000000'; ctx.fill();
      }
      // Zero Right Slanted Polygon
      if (crt < w || crb < w) {
        ctx.beginPath();
        ctx.moveTo(crt, 0); ctx.lineTo(w, 0); ctx.lineTo(w, h); ctx.lineTo(crb, h);
        ctx.closePath(); ctx.fillStyle = '#000000'; ctx.fill();
      }
    }
    ctx.restore();

    // 2. Render Right Clean Retinal Crop Preview (Preserving exact physical aspect ratio)
    if (this.cleanCropPreview) {
      const rightCanvas = this.cleanCropPreview;
      const rCtx = rightCanvas.getContext('2d');
      rightCanvas.width = w;
      rightCanvas.height = h;
      rCtx.clearRect(0, 0, w, h);

      rCtx.save();
      rCtx.fillStyle = '#000000';
      rCtx.fillRect(0, 0, w, h);
      this.traceSmoothPathToCanvas(rCtx, topPts, botPts);
      rCtx.clip();
      rCtx.drawImage(this.imgRaw, 0, 0, w, h);
      // Zero Left Slanted Polygon
      if (clt > 0 || clb > 0) {
        rCtx.beginPath();
        rCtx.moveTo(0, 0); rCtx.lineTo(clt, 0); rCtx.lineTo(clb, h); rCtx.lineTo(0, h);
        rCtx.closePath(); rCtx.fillStyle = '#000000'; rCtx.fill();
      }
      // Zero Right Slanted Polygon
      if (crt < w || crb < w) {
        rCtx.beginPath();
        rCtx.moveTo(crt, 0); rCtx.lineTo(w, 0); rCtx.lineTo(w, h); rCtx.lineTo(crb, h);
        rCtx.closePath(); rCtx.fillStyle = '#000000'; rCtx.fill();
      }
      rCtx.restore();
    }
  }

  toggleViewMode() {
    const modes = ['vectors', 'crop', 'raw', 'mask'];
    const nextIdx = (modes.indexOf(this.viewMode) + 1) % modes.length;
    this.viewMode = modes[nextIdx];

    if (this.viewModeBadge) {
      const labels = {
        vectors: 'VIEW: BOUNDARY VECTORS (ILM + CHOROID)',
        crop: 'VIEW: CLEAN RETINAL CROP (BACKGROUND SUPPRESSED)',
        raw: 'VIEW: RAW UNCROPPED SCAN',
        mask: 'VIEW: BINARY GROUND TRUTH MASK'
      };
      this.viewModeBadge.textContent = labels[this.viewMode];
    }

    if (this.svgOverlay) {
      this.svgOverlay.style.display = (this.viewMode === 'vectors') ? 'block' : 'none';
    }
    if (this.cropPreviewCanvas) {
      this.cropPreviewCanvas.style.display = (this.viewMode === 'crop' || this.viewMode === 'mask') ? 'block' : 'none';
    }
    if (this.imgRaw) {
      this.imgRaw.style.opacity = (this.viewMode === 'crop' || this.viewMode === 'mask') ? '0' : '1';
    }

    const item = this.queue[this.currentIndex];
    if (item) this.renderLiveCropCanvas(item);
  }

  async approveCurrentCrop() {
    if (this.isAnimating) return;
    const item = this.queue[this.currentIndex];
    if (!item) return;

    this.isAnimating = true;

    // 1. Button Pulse & Stamp Pop
    const btnApprove = document.getElementById('btn-swipe-approve');
    if (btnApprove) {
      btnApprove.classList.add('btn-pulse-approve');
      setTimeout(() => btnApprove.classList.remove('btn-pulse-approve'), 200);
    }
    if (this.stampApprove) this.stampApprove.classList.add('stamp-active-approve');

    // 2. Card Exit Animation (Slide Right with Tilt)
    if (this.card) {
      this.card.style.transition = 'transform 0.24s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.20s ease';
      this.card.style.transform = 'translateX(480px) rotate(16deg) scale(0.94)';
      this.card.style.opacity = '0';
    }

    // 3. Flash preview wrapper
    const preview = document.querySelector('.swipe-preview-wrapper');
    if (preview) {
      preview.classList.add('preview-flash');
      setTimeout(() => preview.classList.remove('preview-flash'), 300);
    }

    // 4. Background save call with 2-point slanted lateral bounds
    saveCuratedCrop(
      item.folder,
      item.filename,
      item.y_top_points,
      item.y_bot_points,
      item.crop_left || 0,
      item.crop_right || item.width,
      item.crop_left_top !== undefined ? item.crop_left_top : (item.crop_left || 0),
      item.crop_left_bot !== undefined ? item.crop_left_bot : (item.crop_left || 0),
      item.crop_right_top !== undefined ? item.crop_right_top : (item.crop_right || item.width),
      item.crop_right_bot !== undefined ? item.crop_right_bot : (item.crop_right || item.width)
    ).catch(err => {
      console.error('Error saving curated crop:', err);
    });

    if (!item.is_curated) {
      item.is_curated = true;
      if (this.stats && this.stats.total_curated !== undefined) {
        this.stats.total_curated++;
        if (this.stats.folders && this.stats.folders[item.folder]) {
          this.stats.folders[item.folder].curated++;
        }
      }
    }
    this.scheduleRefreshStats(600);

    // 5. Entrance transition after exit completes
    setTimeout(() => {
      if (this.stampApprove) this.stampApprove.classList.remove('stamp-active-approve');
      this.currentIndex++;

      if (this.card) {
        this.card.style.transition = 'none';
        this.card.style.transform = 'translateX(-60px) scale(0.96)';
        this.card.style.opacity = '0';
      }

      if (this.currentIndex >= this.queue.length) {
        this.loadQueue(this.currentIndex);
        this.isAnimating = false;
      } else {
        this.renderCurrentScan();
        requestAnimationFrame(() => {
          if (this.card) {
            this.card.offsetHeight;
            this.card.style.transition = 'transform 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.22s ease';
            this.card.style.transform = 'translateX(0px) rotate(0deg) scale(1.0)';
            this.card.style.opacity = '1';
          }
          setTimeout(() => {
            this.isAnimating = false;
          }, 280);
        });
      }
    }, 180);
  }

  skipCurrentCrop() {
    if (this.isAnimating) return;
    const item = this.queue[this.currentIndex];
    if (!item) return;

    this.isAnimating = true;

    // 1. Button Pulse & Stamp Pop
    const btnSkip = document.getElementById('btn-swipe-skip');
    if (btnSkip) {
      btnSkip.classList.add('btn-pulse-skip');
      setTimeout(() => btnSkip.classList.remove('btn-pulse-skip'), 200);
    }
    if (this.stampSkip) this.stampSkip.classList.add('stamp-active-skip');

    // 2. Card Exit Animation (Slide Left with Tilt)
    if (this.card) {
      this.card.style.transition = 'transform 0.24s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.20s ease';
      this.card.style.transform = 'translateX(-480px) rotate(-16deg) scale(0.94)';
      this.card.style.opacity = '0';
    }

    // 3. Entrance transition after exit completes
    setTimeout(() => {
      if (this.stampSkip) this.stampSkip.classList.remove('stamp-active-skip');
      this.currentIndex++;

      if (this.card) {
        this.card.style.transition = 'none';
        this.card.style.transform = 'translateX(60px) scale(0.96)';
        this.card.style.opacity = '0';
      }

      if (this.currentIndex >= this.queue.length) {
        this.loadQueue(this.currentIndex);
        this.isAnimating = false;
      } else {
        this.renderCurrentScan();
        requestAnimationFrame(() => {
          if (this.card) {
            this.card.offsetHeight;
            this.card.style.transition = 'transform 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.22s ease';
            this.card.style.transform = 'translateX(0px) rotate(0deg) scale(1.0)';
            this.card.style.opacity = '1';
          }
          setTimeout(() => {
            this.isAnimating = false;
          }, 280);
        });
      }
    }, 180);
  }

  stepPrevious() {
    if (this.isAnimating || this.currentIndex <= 0) return;
    this.isAnimating = true;

    if (this.card) {
      this.card.style.transition = 'transform 0.20s ease, opacity 0.18s ease';
      this.card.style.transform = 'translateX(100px) scale(0.96)';
      this.card.style.opacity = '0';
    }

    setTimeout(() => {
      this.currentIndex--;
      if (this.card) {
        this.card.style.transition = 'none';
        this.card.style.transform = 'translateX(-100px) scale(0.96)';
        this.card.style.opacity = '0';
      }
      this.renderCurrentScan();
      requestAnimationFrame(() => {
        if (this.card) {
          this.card.offsetHeight;
          this.card.style.transition = 'transform 0.26s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.20s ease';
          this.card.style.transform = 'translateX(0px) scale(1.0)';
          this.card.style.opacity = '1';
        }
        setTimeout(() => {
          this.isAnimating = false;
        }, 260);
      });
    }, 150);
  }

  renderEmptyQueue() {
    if (this.sampleMetaEl) this.sampleMetaEl.textContent = 'All scans in this category have been filtered/curated!';
    if (this.svgOverlay) this.svgOverlay.innerHTML = '';
  }

  openRetrainModal() {
    if (this.retrainModal) {
      this.updateRetrainDatasetStats();
      this.retrainModal.style.display = 'flex';
      if (this.backgroundHeartbeatInterval) {
        clearInterval(this.backgroundHeartbeatInterval);
        this.backgroundHeartbeatInterval = null;
      }
      this.pollRetrainStatus();
    }
  }

  updateRetrainDatasetStats() {
    const descEl = document.getElementById('retrain-desc-text');
    if (!descEl) return;
    const curatedCount = (this.stats && this.stats.total_curated !== undefined) ? this.stats.total_curated : 0;
    const baseCount = 5719;
    const totalCount = baseCount + curatedCount;

    if (curatedCount === 0) {
      descEl.innerHTML = `Fine-tune the Attention U-Net on all <strong>${baseCount.toLocaleString()} base segmentation pairs</strong> (0 newly approved Classified-masked/ pairs curated yet) using Apple Silicon MPS hardware acceleration.`;
    } else {
      descEl.innerHTML = `Fine-tune the Attention U-Net on all <strong>${baseCount.toLocaleString()} base segmentation pairs + <span style="color:#10b981;font-weight:700;">${curatedCount.toLocaleString()} newly approved Classified-masked/ pairs</span></strong> (<strong style="color:#00f2fe;">${totalCount.toLocaleString()} total pairs</strong>) using Apple Silicon MPS hardware acceleration.`;
    }
  }

  closeRetrainModal() {
    if (this.retrainModal) {
      this.retrainModal.style.display = 'none';
      if (this.retrainPollInterval) {
        clearInterval(this.retrainPollInterval);
        this.retrainPollInterval = null;
      }
      this.startBackgroundHeartbeat();
    }
  }

  startBackgroundHeartbeat() {
    if (this.backgroundHeartbeatInterval) return;
    this.backgroundHeartbeatInterval = setInterval(async () => {
      try {
        const status = await fetchRetrainStatus();
        this.updateRetrainUI(status);
        if (!status.running) {
          clearInterval(this.backgroundHeartbeatInterval);
          this.backgroundHeartbeatInterval = null;
        }
      } catch (e) {
        console.warn('Background heartbeat error:', e);
      }
    }, 3000);
  }

  async startRetraining() {
    try {
      const epochsInput = document.getElementById('retrain-epochs-input');
      const epochs = epochsInput ? parseInt(epochsInput.value, 10) : 10;
      await triggerUNetRetrain(epochs, 2e-4);
      if (this.btnStartRetrainExec) this.btnStartRetrainExec.style.display = 'none';
      if (this.btnStopRetrainExec) {
        this.btnStopRetrainExec.style.display = 'block';
        this.btnStopRetrainExec.textContent = 'Stop Retraining';
        this.btnStopRetrainExec.disabled = false;
      }
      this.pollRetrainStatus();
    } catch (e) {
      alert(`Failed to start retraining: ${e.message}`);
    }
  }

  async stopRetraining() {
    if (!confirm('Are you sure you want to halt retraining? The current best checkpoint will be safely preserved.')) {
      return;
    }
    try {
      if (this.btnStopRetrainExec) {
        this.btnStopRetrainExec.textContent = 'Stopping...';
        this.btnStopRetrainExec.disabled = true;
      }
      const res = await stopUNetRetraining();
      if (this.retrainMsgEl) {
        this.retrainMsgEl.textContent = res.message || 'Stop requested. Finishing current batch...';
      }
    } catch (e) {
      alert(`Failed to stop retraining: ${e.message}`);
      if (this.btnStopRetrainExec) {
        this.btnStopRetrainExec.textContent = 'Stop Retraining';
        this.btnStopRetrainExec.disabled = false;
      }
    }
  }

  updateRetrainUI(status) {
    if (!status) return;

    let compositePct = 0;
    if (status.total_epochs > 0) {
      const epochIdx = Math.max(0, (status.epoch || 1) - 1);
      const batchFraction = (status.total_batches > 0) ? (status.batch / status.total_batches) : 0;
      compositePct = Math.min(100, Math.max(0, ((epochIdx + batchFraction) / status.total_epochs) * 100));
    }
    if (status.phase === 'complete') compositePct = 100;

    if (this.retrainMsgEl) this.retrainMsgEl.textContent = status.message || 'Ready';
    if (this.retrainProgressBar) this.retrainProgressBar.style.width = `${compositePct.toFixed(1)}%`;
    if (this.retrainPctBadge) this.retrainPctBadge.textContent = `${compositePct.toFixed(1)}%`;

    // Toggle Start vs Stop button visibility based on running state
    if (status.running) {
      if (this.btnStartRetrainExec) this.btnStartRetrainExec.style.display = 'none';
      if (this.btnStopRetrainExec) {
        this.btnStopRetrainExec.style.display = 'block';
        if (!this.btnStopRetrainExec.disabled) {
          this.btnStopRetrainExec.textContent = 'Stop Retraining';
        }
      }
    } else {
      if (this.btnStartRetrainExec) this.btnStartRetrainExec.style.display = 'block';
      if (this.btnStopRetrainExec) {
        this.btnStopRetrainExec.style.display = 'none';
        this.btnStopRetrainExec.textContent = 'Stop Retraining';
        this.btnStopRetrainExec.disabled = false;
      }
    }

    // Dynamic Retrain Button internal progress bar & state
    if (this.btnTriggerRetrain) {
      if (status.running) {
        this.btnTriggerRetrain.classList.add('training-active');
        if (this.btnRetrainFill) this.btnRetrainFill.style.width = `${compositePct.toFixed(1)}%`;
        if (this.btnRetrainShimmer) this.btnRetrainShimmer.style.display = 'block';
        if (this.btnRetrainText) {
          this.btnRetrainText.textContent = `U-Net Training In Progress (${compositePct.toFixed(0)}%)`;
        }
      } else if (status.phase === 'complete') {
        this.btnTriggerRetrain.classList.remove('training-active');
        if (this.btnRetrainFill) this.btnRetrainFill.style.width = '100%';
        if (this.btnRetrainShimmer) this.btnRetrainShimmer.style.display = 'none';
        if (this.btnRetrainText) {
          this.btnRetrainText.textContent = `U-Net Retrained (Dice: ${(status.val_dice * 100).toFixed(1)}%)`;
        }
      } else {
        this.btnTriggerRetrain.classList.remove('training-active');
        if (this.btnRetrainFill) this.btnRetrainFill.style.width = '0%';
        if (this.btnRetrainShimmer) this.btnRetrainShimmer.style.display = 'none';
        if (this.btnRetrainText) {
          this.btnRetrainText.textContent = 'Retrain U-Net on Curated Data';
        }
      }
    }

    if (this.retrainMetricsEl) {
      const batchProgress = (status.total_batches > 0)
        ? `${status.batch} / ${status.total_batches} <span style="color:#64748b;">(${((status.batch / status.total_batches) * 100).toFixed(0)}%)</span>`
        : '—';
      const batchLossStr = (status.batch_loss > 0) ? status.batch_loss.toFixed(4) : '—';
      const trainLossStr = (status.train_loss > 0) ? status.train_loss.toFixed(4) : '—';
      const valLossStr = (status.val_loss > 0) ? status.val_loss.toFixed(4) : '—';
      const diceStr = (status.val_dice > 0) ? `${(status.val_dice * 100).toFixed(2)}%` : '0.00%';

      this.retrainMetricsEl.innerHTML = `
        <div><strong>Epoch:</strong> ${status.epoch} / ${status.total_epochs}</div>
        <div><strong>Batch Progress:</strong> ${batchProgress}</div>
        <div><strong>Live Batch Loss:</strong> <span style="color:#38bdf8;font-weight:600;">${batchLossStr}</span></div>
        <div><strong>Train Loss (Avg):</strong> ${trainLossStr}</div>
        <div><strong>Val Loss:</strong> ${valLossStr}</div>
        <div><strong>Val Dice (F1):</strong> <span style="color:#10b981;font-weight:700;">${diceStr}</span></div>
      `;
    }
  }

  pollRetrainStatus() {
    if (this.retrainPollInterval) clearInterval(this.retrainPollInterval);

    const update = async () => {
      try {
        const status = await fetchRetrainStatus();
        this.updateRetrainUI(status);

        if (!status.running && status.epoch > 0) {
          clearInterval(this.retrainPollInterval);
          this.retrainPollInterval = null;
          this.refreshStats();
          this.loadQueue(0, true);
        }
      } catch (e) {
        console.warn('Retrain poll error:', e);
      }
    };

    update();
    this.retrainPollInterval = setInterval(update, 600);
  }

  async openHistoryModal() {
    if (this.modalModelHistory) {
      this.modalModelHistory.style.display = 'flex';
      await this.renderTrainingHistory();
    }
  }

  closeHistoryModal() {
    if (this.modalModelHistory) {
      this.modalModelHistory.style.display = 'none';
    }
  }

  async renderTrainingHistory() {
    if (!this.historyRoundsList) return;
    this.historyRoundsList.innerHTML = '<div style="color: #64748b; font-size: 0.85rem; text-align: center; padding: 20px;">Loading training history...</div>';

    try {
      const data = await fetchTrainingHistory();
      if (!data || !data.rounds) {
        this.historyRoundsList.innerHTML = '<div style="color: #ef4444; font-size: 0.85rem; text-align: center; padding: 20px;">No training history records found.</div>';
        return;
      }

      // Update header summary stats
      if (this.historyStatDice) {
        const bestDice = data.current_best_val_dice || (data.rounds.length > 0 ? Math.max(...data.rounds.map(r => r.best_val_dice || 0)) : 0);
        this.historyStatDice.textContent = `${(bestDice * 100).toFixed(2)}%`;
      }
      if (this.historyStatIou) {
        const bestIou = data.rounds.length > 0 ? Math.max(...data.rounds.map(r => r.best_val_iou || 0)) : 0;
        this.historyStatIou.textContent = `${(bestIou * 100).toFixed(2)}%`;
      }
      if (this.historyStatRounds) {
        this.historyStatRounds.textContent = data.rounds.length;
      }
      if (this.historyTotalRoundsBadge) {
        this.historyTotalRoundsBadge.textContent = `${data.rounds.length} ${data.rounds.length === 1 ? 'Round' : 'Rounds'}`;
      }
      if (this.historyStatCurated) {
        const curated = (this.stats && this.stats.total_curated !== undefined) ? this.stats.total_curated : (data.rounds[data.rounds.length - 1]?.dataset?.curated_samples || 0);
        this.historyStatCurated.textContent = curated.toLocaleString();
      }

      // Build rounds list (reverse order so newest rounds appear at top)
      const reversedRounds = [...data.rounds].reverse();
      let html = '';

      reversedRounds.forEach((round) => {
        const isScratch = (round.mode === 'From Scratch' || round.round_id === 1);
        const modeBadge = isScratch
          ? '<span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">From Scratch</span>'
          : '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Fine-Tuning</span>';

        let statusBadge = '';
        if (round.user_stopped) {
          statusBadge = '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Halted by User</span>';
        } else if (round.early_stopped) {
          statusBadge = '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Early Stopped</span>';
        } else {
          statusBadge = '<span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Completed</span>';
        }

        const dateStr = round.completed_at ? new Date(round.completed_at).toLocaleString() : (round.started_at ? new Date(round.started_at).toLocaleString() : '—');
        const durationStr = round.duration_minutes ? `${round.duration_minutes.toFixed(1)} mins` : '—';
        const bestDiceStr = round.best_val_dice ? `${(round.best_val_dice * 100).toFixed(2)}%` : '—';
        const bestIouStr = round.best_val_iou ? `${(round.best_val_iou * 100).toFixed(2)}%` : '—';
        const bestLossStr = round.best_val_loss ? round.best_val_loss.toFixed(4) : '—';
        const epochsCompleted = round.epochs_completed || (round.epoch_logs ? round.epoch_logs.length : 0);
        const epochsPlanned = round.epochs_planned || epochsCompleted;
        const totalTrain = round.dataset?.total_train?.toLocaleString() || '—';
        const totalVal = round.dataset?.total_val?.toLocaleString() || '—';
        const curatedCount = round.dataset?.curated_samples?.toLocaleString() || '0';

        // Build epoch rows table if epoch_logs exist
        let epochTableHtml = '';
        if (round.epoch_logs && round.epoch_logs.length > 0) {
          const rows = round.epoch_logs.map(log => `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); text-align: left; font-size: 0.75rem;">
              <td style="padding: 6px 10px; color: #f1f5f9; font-weight: 600;">#${log.epoch}</td>
              <td style="padding: 6px 10px; color: #94a3b8;">${log.train_loss ? log.train_loss.toFixed(4) : '—'}</td>
              <td style="padding: 6px 10px; color: #94a3b8;">${log.val_loss ? log.val_loss.toFixed(4) : '—'}</td>
              <td style="padding: 6px 10px; color: #10b981; font-weight: 600;">${log.val_dice ? (log.val_dice * 100).toFixed(2) + '%' : '—'}</td>
              <td style="padding: 6px 10px; color: #38bdf8; font-weight: 600;">${log.val_iou ? (log.val_iou * 100).toFixed(2) + '%' : '—'}</td>
              <td style="padding: 6px 10px; color: #64748b; font-family: monospace;">${log.learning_rate ? log.learning_rate.toExponential(2) : '—'}</td>
            </tr>
          `).join('');

          epochTableHtml = `
            <details style="margin-top: 10px; background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 6px 10px;">
              <summary style="font-size: 0.75rem; color: #93c5fd; cursor: pointer; user-select: none; font-weight: 600;">
                View Per-Epoch Progression Log (${round.epoch_logs.length} epochs)
              </summary>
              <div style="overflow-x: auto; margin-top: 8px;">
                <table style="width: 100%; border-collapse: collapse;">
                  <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; font-size: 0.70rem; color: #64748b; text-transform: uppercase;">
                      <th style="padding: 6px 10px;">Epoch</th>
                      <th style="padding: 6px 10px;">Train Loss</th>
                      <th style="padding: 6px 10px;">Val Loss</th>
                      <th style="padding: 6px 10px;">Val Dice</th>
                      <th style="padding: 6px 10px;">Val IoU</th>
                      <th style="padding: 6px 10px;">LR</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${rows}
                  </tbody>
                </table>
              </div>
            </details>
          `;
        }

        html += `
          <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.90rem; font-weight: 700; color: #f1f5f9;">Round ${round.round_id}</span>
                ${modeBadge}
                ${statusBadge}
              </div>
              <span style="font-size: 0.72rem; color: #64748b;">${dateStr}</span>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 0.78rem; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; margin-top: 4px;">
              <div>
                <span style="color: #64748b; font-size: 0.70rem; text-transform: uppercase; display: block;">Best Val Dice</span>
                <strong style="color: #10b981; font-size: 0.92rem;">${bestDiceStr}</strong>
              </div>
              <div>
                <span style="color: #64748b; font-size: 0.70rem; text-transform: uppercase; display: block;">Best Val IoU</span>
                <strong style="color: #38bdf8; font-size: 0.92rem;">${bestIouStr}</strong>
              </div>
              <div>
                <span style="color: #64748b; font-size: 0.70rem; text-transform: uppercase; display: block;">Best Val Loss</span>
                <strong style="color: #f1f5f9; font-size: 0.92rem;">${bestLossStr}</strong>
              </div>
              <div>
                <span style="color: #64748b; font-size: 0.70rem; text-transform: uppercase; display: block;">Epochs / Time</span>
                <strong style="color: #f1f5f9;">${epochsCompleted} / ${epochsPlanned}</strong> <span style="color: #64748b; font-size: 0.72rem;">(${durationStr})</span>
              </div>
            </div>

            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.74rem; color: #94a3b8; padding: 0 4px;">
              <div>Dataset: <strong>${totalTrain}</strong> train pairs, <strong>${totalVal}</strong> val pairs (including <strong>${curatedCount}</strong> curated)</div>
              <div>Hardware: <code style="color: #a855f7;">${round.device || 'mps'}</code></div>
            </div>

            ${epochTableHtml}
          </div>
        `;
      });

      this.historyRoundsList.innerHTML = html;
    } catch (e) {
      console.error('Failed to render training history:', e);
      this.historyRoundsList.innerHTML = `<div style="color: #ef4444; font-size: 0.85rem; text-align: center; padding: 20px;">Failed to load training history: ${e.message}</div>`;
    }
  }
}

// Global instance
window.swipingStudio = new SwipingStudio();
