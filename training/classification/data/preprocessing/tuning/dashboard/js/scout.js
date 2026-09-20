/**
 * training/classification/data/preprocessing/tuning/dashboard/js/scout.js
 *
 * Scouting File Manager & Interactive Split Viewer for Classified-unet-masked.
 * Provides Side-by-Side, Comparison Slider Curtain, and Overlay Blend modes
 * with prominent red-background warning treatment for suspicious images.
 */

class ScoutExplorer {
  constructor() {
    this.active = false;
    this.folder = '';
    this.filterType = 'all'; // 'all', 'suspicious', 'clean'
    this.searchQuery = '';
    this.sortBy = 'name';
    this.images = [];
    this.selectedIndex = -1;
    this.selectedScan = null;
    this.viewMode = 'split-side'; // 'split-side', 'split-slider', 'overlay'
    this.sliderPct = 50;
    this.overlayOpacity = 0.65;
    this.isDraggingSlider = false;

    // DOM references
    this.container = document.getElementById('scout-explorer-view');
    this.totalCountEl = document.getElementById('scout-total-count');
    this.cleanCountEl = document.getElementById('scout-clean-count');
    this.suspiciousCountEl = document.getElementById('scout-suspicious-count');
    this.folderSelect = document.getElementById('scout-folder-select');
    this.searchInput = document.getElementById('scout-search-input');
    this.filterBtns = document.querySelectorAll('.scout-filter-btn');
    this.listEl = document.getElementById('scout-fm-list');
    this.stageViewport = document.getElementById('scout-stage-viewport');
    this.anomalyBanner = document.getElementById('scout-anomaly-banner');
    this.anomalyFlagsEl = document.getElementById('scout-anomaly-flags');
    this.viewerFilename = document.getElementById('scout-viewer-filename');
    this.viewerBadge = document.getElementById('scout-viewer-badge');
    this.viewModeBtns = document.querySelectorAll('.scout-mode-btn');

    // View containers
    this.splitSideContainer = document.getElementById('scout-split-side');
    this.splitSliderContainer = document.getElementById('scout-split-slider');
    this.overlayContainer = document.getElementById('scout-overlay-view');

    // Images
    this.realImgSide = document.getElementById('scout-real-img-side');
    this.maskImgSide = document.getElementById('scout-mask-img-side');
    this.realImgSlider = document.getElementById('scout-real-img-slider');
    this.maskImgSlider = document.getElementById('scout-mask-img-slider');
    this.realImgOverlay = document.getElementById('scout-real-img-overlay');
    this.maskImgOverlay = document.getElementById('scout-mask-img-overlay');

    // Crosshairs
    this.canvasSideReal = document.getElementById('scout-canvas-side-real');
    this.canvasSideMask = document.getElementById('scout-canvas-side-mask');

    // Drawer metrics
    this.mAreaRatio = document.getElementById('scout-m-area');
    this.mThickness = document.getElementById('scout-m-thickness');
    this.mComponents = document.getElementById('scout-m-components');
    this.mDerivative = document.getElementById('scout-m-derivative');
    this.mMargin = document.getElementById('scout-m-margin');

    // Surgical BG Tools (Eraser & U-Net Window Rerun) & Undo
    this.activeTool = 'eraser'; // 'eraser' or 'unet-window'
    this.isToolActive = false;
    this.canUndo = false;
    this.isExecutingAction = false;
    this.toolSelect = document.getElementById('scout-active-tool-select');
    this.toolActionBtn = document.getElementById('scout-tool-action-btn');
    this.toolActionFill = document.getElementById('scout-ablate-fill');
    this.toolActionShimmer = document.getElementById('scout-ablate-shimmer');
    this.toolActionText = document.getElementById('scout-ablate-text');
    this.btnUndo = document.getElementById('scout-btn-undo');
    this.toolStatusEl = document.getElementById('scout-tool-status');

    // Drag-selection window box state for U-Net rerun
    this.isBoxSelecting = false;
    this.boxStartPoint = null;
    this.currentBox = null; // { x1, y1, x2, y2 }

    this.bindEvents();
  }

  bindEvents() {
    if (this.folderSelect) {
      this.folderSelect.addEventListener('change', () => {
        this.folder = this.folderSelect.value;
        this.loadImages();
      });
    }

    if (this.searchInput) {
      let debounceTimer = null;
      this.searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          this.searchQuery = this.searchInput.value.trim();
          this.loadImages();
        }, 250);
      });
    }

    this.filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.filterType = btn.dataset.filter || 'all';
        this.loadImages();
      });
    });

    this.viewModeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.viewModeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.setViewMode(btn.dataset.mode || 'split-side');
      });
    });

    // Comparison slider drag handling
    if (this.splitSliderContainer) {
      const updateSliderFromEvent = (e) => {
        const rect = this.splitSliderContainer.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const offsetX = Math.max(0, Math.min(rect.width, clientX - rect.left));
        this.sliderPct = Math.round((offsetX / rect.width) * 100);
        this.splitSliderContainer.style.setProperty('--split-pos', `${this.sliderPct}%`);
      };

      this.splitSliderContainer.addEventListener('mousedown', (e) => {
        this.isDraggingSlider = true;
        updateSliderFromEvent(e);
      });

      window.addEventListener('mousemove', (e) => {
        if (this.isDraggingSlider) {
          updateSliderFromEvent(e);
        }
      });

      window.addEventListener('mouseup', () => {
        this.isDraggingSlider = false;
      });

      this.splitSliderContainer.addEventListener('touchstart', (e) => {
        this.isDraggingSlider = true;
        updateSliderFromEvent(e);
      }, { passive: true });

      window.addEventListener('touchmove', (e) => {
        if (this.isDraggingSlider) {
          updateSliderFromEvent(e);
        }
      }, { passive: true });

      window.addEventListener('touchend', () => {
        this.isDraggingSlider = false;
      });
    }

    // Tool Selector Dropdown
    if (this.toolSelect) {
      this.toolSelect.addEventListener('change', () => {
        this.setToolType(this.toolSelect.value);
      });
    }

    // Tool Action Button Toggle
    if (this.toolActionBtn) {
      this.toolActionBtn.addEventListener('click', () => {
        this.toggleActiveTool();
      });
    }

    if (this.btnUndo) {
      this.btnUndo.addEventListener('click', () => {
        this.undoLastAction();
      });
    }

    // Zero-drift coordinate calculation helper
    const getExactImageCoords = (canvas, e) => {
      const rect = canvas.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      const imgX = Math.min(canvas.width - 1, Math.floor(relX * canvas.width));
      const imgY = Math.min(canvas.height - 1, Math.floor(relY * canvas.height));
      return { x: imgX, y: imgY };
    };

    // Synchronized crosshair cursors, surgical reticle, and drag box on side-by-side mode
    const syncCrosshair = (sourceCanvas, targetCanvas, e) => {
      if (!sourceCanvas || !targetCanvas) return;
      const { x, y } = getExactImageCoords(sourceCanvas, e);

      [sourceCanvas, targetCanvas].forEach(canvas => {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (this.isToolActive && this.activeTool === 'eraser') {
          // Surgical Eraser Reticle
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 2]);

          // Draw target circle at click focus (radius 18px on canvas)
          ctx.beginPath();
          ctx.arc(x, y, 18, 0, 2 * Math.PI);
          ctx.stroke();

          // Center crosshair inside circle
          ctx.setLineDash([]);
          ctx.strokeStyle = '#f87171';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x - 24, y);
          ctx.lineTo(x + 24, y);
          ctx.moveTo(x, y - 24);
          ctx.lineTo(x, y + 24);
          ctx.stroke();

          // Display surgical coordinates & prompt
          ctx.fillStyle = '#f87171';
          ctx.font = 'bold 11px monospace';
          ctx.fillText(`ERASER FOCUS [${x}, ${y}]`, 8, canvas.height - 8);
        } else if (this.isToolActive && this.activeTool === 'unet-window') {
          // U-Net Window Selection Reticle or Active Drag Box
          if (this.isBoxSelecting && this.boxStartPoint) {
            const bx = Math.min(this.boxStartPoint.x, x);
            const by = Math.min(this.boxStartPoint.y, y);
            const bw = Math.abs(x - this.boxStartPoint.x);
            const bh = Math.abs(y - this.boxStartPoint.y);

            // Shaded window box
            ctx.fillStyle = 'rgba(59, 130, 246, 0.22)';
            ctx.fillRect(bx, by, bw, bh);

            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 2]);
            ctx.strokeRect(bx, by, bw, bh);

            ctx.setLineDash([]);
            ctx.fillStyle = '#60a5fa';
            ctx.font = 'bold 11px monospace';
            ctx.fillText(`U-NET WINDOW: ${bw}x${bh} px [${bx}, ${by}]`, Math.max(8, bx), Math.max(16, by - 6));
          } else {
            // Precision corner reticle before dragging
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);

            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();

            ctx.setLineDash([]);
            ctx.fillStyle = '#60a5fa';
            ctx.font = 'bold 11px monospace';
            ctx.fillText(`U-NET CORNER [${x}, ${y}] - DRAG BOX`, 8, canvas.height - 8);
          }
        } else {
          // Standard inspection crosshairs
          ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 3]);

          // Draw cross lines
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvas.height);
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();

          // Draw coordinate text
          ctx.fillStyle = '#00f2fe';
          ctx.font = '11px monospace';
          ctx.fillText(`X:${x} Y:${y}`, 8, canvas.height - 8);
        }
      });
    };

    const clearCrosshair = (c1, c2) => {
      if (this.isBoxSelecting) return;
      if (c1) c1.getContext('2d').clearRect(0, 0, c1.width, c1.height);
      if (c2) c2.getContext('2d').clearRect(0, 0, c2.width, c2.height);
    };

    const handleCanvasMouseDown = (canvas, e) => {
      if (!this.isToolActive || this.isExecutingAction || !this.selectedScan) return;
      const coords = getExactImageCoords(canvas, e);

      if (this.activeTool === 'unet-window') {
        this.isBoxSelecting = true;
        this.boxStartPoint = coords;
      }
    };

    const handleCanvasMouseUp = (canvas, e) => {
      if (!this.isToolActive || this.isExecutingAction || !this.selectedScan) return;
      const coords = getExactImageCoords(canvas, e);

      if (this.activeTool === 'unet-window' && this.isBoxSelecting && this.boxStartPoint) {
        this.isBoxSelecting = false;
        const x1 = this.boxStartPoint.x;
        const y1 = this.boxStartPoint.y;
        const x2 = coords.x;
        const y2 = coords.y;
        this.boxStartPoint = null;

        // Clear canvas drawings
        clearCrosshair(this.canvasSideReal, this.canvasSideMask);

        const w = Math.abs(x2 - x1);
        const h = Math.abs(y2 - y1);
        if (w >= 10 && h >= 10) {
          this.executeUnetWindow(x1, y1, x2, y2);
        }
      } else if (this.activeTool === 'eraser') {
        this.executeAblation(coords.x, coords.y);
      }
    };

    if (this.canvasSideReal && this.canvasSideMask) {
      [this.canvasSideReal, this.canvasSideMask].forEach(canvas => {
        const otherCanvas = (canvas === this.canvasSideReal) ? this.canvasSideMask : this.canvasSideReal;

        canvas.addEventListener('mousemove', (e) => {
          syncCrosshair(canvas, otherCanvas, e);
        });
        canvas.addEventListener('mouseleave', () => {
          clearCrosshair(canvas, otherCanvas);
        });
        canvas.addEventListener('mousedown', (e) => {
          handleCanvasMouseDown(canvas, e);
        });
        canvas.addEventListener('mouseup', (e) => {
          handleCanvasMouseUp(canvas, e);
        });
      });
    }

    // Keyboard navigation (Left/Right or J/K) and Ctrl+Z Undo
    window.addEventListener('keydown', (e) => {
      if (!this.active) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        if (this.canUndo && !this.isAblating) {
          this.undoAblation();
        }
        return;
      }

      if (e.key === 'Escape' && this.isAblateActive) {
        this.toggleAblateTool(false);
        return;
      }

      if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {
        this.selectPrevious();
      } else if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {
        this.selectNext();
      }
    });
  }

  async activate() {
    this.active = true;
    if (this.container) {
      this.container.style.display = 'flex';
    }
    await this.loadOverview();
    await this.loadFolders();
    await this.loadImages();
  }

  deactivate() {
    this.active = false;
    if (this.container) {
      this.container.style.display = 'none';
    }
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (this.splitSideContainer) {
      this.splitSideContainer.style.display = (mode === 'split-side') ? 'grid' : 'none';
    }
    if (this.splitSliderContainer) {
      this.splitSliderContainer.style.display = (mode === 'split-slider') ? 'block' : 'none';
    }
    if (this.overlayContainer) {
      this.overlayContainer.style.display = (mode === 'overlay') ? 'block' : 'none';
    }
  }

  async loadOverview() {
    try {
      const res = await fetch('/api/scout/overview');
      if (!res.ok) return;
      const data = await res.json();
      if (this.totalCountEl) this.totalCountEl.textContent = `${data.total_scans} scans`;
      if (this.cleanCountEl) this.cleanCountEl.textContent = `${data.clean_count} clean`;
      if (this.suspiciousCountEl) {
        this.suspiciousCountEl.textContent = `${data.suspicious_count} suspicious`;
      }
    } catch (err) {
      console.error('Failed to load scout overview:', err);
    }
  }

  async loadFolders() {
    try {
      const res = await fetch('/api/scout/tree');
      if (!res.ok) return;
      const folders = await res.json();
      if (!this.folderSelect) return;

      const currentVal = this.folderSelect.value;
      this.folderSelect.innerHTML = '<option value="">All Categories & Folders</option>';
      folders.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.folder;
        const suspTag = item.suspicious > 0 ? ` [${item.suspicious} SUSPICIOUS]` : '';
        opt.textContent = `${item.folder || 'Root'} (${item.total}${suspTag})`;
        this.folderSelect.appendChild(opt);
      });
      this.folderSelect.value = currentVal;
    } catch (err) {
      console.error('Failed to load scout folders:', err);
    }
  }

  async loadImages() {
    try {
      if (this.listEl) {
        this.listEl.innerHTML = '<div style="color: #64748b; font-size: 0.8rem; padding: 12px;">Loading scans...</div>';
      }

      const params = new URLSearchParams();
      if (this.folder) params.set('folder', this.folder);
      if (this.filterType && this.filterType !== 'all') params.set('filter', this.filterType);
      if (this.searchQuery) params.set('query', this.searchQuery);
      params.set('limit', '100');

      const res = await fetch(`/api/scout/images?${params.toString()}`);
      if (!res.ok) throw new Error('API request failed');
      const data = await res.json();

      this.images = data.images || [];
      this.renderList();

      if (this.images.length > 0) {
        this.selectImage(0);
      } else {
        this.clearViewer();
      }
    } catch (err) {
      if (this.listEl) {
        this.listEl.innerHTML = `<div style="color: #ef4444; font-size: 0.8rem; padding: 12px;">Error: ${err.message}</div>`;
      }
    }
  }

  renderList() {
    if (!this.listEl) return;
    this.listEl.innerHTML = '';

    if (this.images.length === 0) {
      this.listEl.innerHTML = '<div style="color: #94a3b8; font-size: 0.8rem; padding: 16px; text-align: center;">No scans match criteria.</div>';
      return;
    }

    this.images.forEach((scan, idx) => {
      const card = document.createElement('div');
      const isSuspicious = scan.is_suspicious || scan.severity === 'WARNING' || scan.severity === 'CRITICAL';

      // KEY REQUIREMENT: Suspicious images will have a red background
      card.className = `scout-card ${isSuspicious ? 'suspicious' : 'clean'} ${idx === this.selectedIndex ? 'active' : ''}`;
      card.dataset.index = idx;

      const badgeClass = scan.severity === 'CRITICAL' ? 'critical' : (scan.severity === 'WARNING' ? 'warning' : 'clean');

      card.innerHTML = `
        <img class="scout-card-thumb" src="${scan.image_url}" loading="lazy" alt="thumb">
        <div class="scout-card-info">
          <div class="scout-card-title" title="${scan.filename}">${scan.filename}</div>
          <div class="scout-card-meta">${scan.folder || 'Root'}</div>
        </div>
        <span class="scout-card-badge ${badgeClass}">${scan.severity}</span>
      `;

      card.addEventListener('click', () => {
        this.selectImage(idx);
      });

      this.listEl.appendChild(card);
    });
  }

  selectPrevious() {
    if (this.selectedIndex > 0) {
      this.selectImage(this.selectedIndex - 1);
    }
  }

  selectNext() {
    if (this.selectedIndex < this.images.length - 1) {
      this.selectImage(this.selectedIndex + 1);
    }
  }

  async selectImage(index) {
    if (index < 0 || index >= this.images.length) return;
    this.selectedIndex = index;
    const scan = this.images[index];
    this.selectedScan = scan;

    // Update active highlight in sidebar list
    const cards = this.listEl.querySelectorAll('.scout-card');
    cards.forEach((c, idx) => {
      if (idx === index) {
        c.classList.add('active');
        c.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        c.classList.remove('active');
      }
    });

    const isSuspicious = scan.is_suspicious || scan.severity === 'WARNING' || scan.severity === 'CRITICAL';

    // KEY REQUIREMENT: Suspicious images will have a red background in the stage viewport
    if (this.stageViewport) {
      if (isSuspicious) {
        this.stageViewport.classList.add('suspicious-active');
      } else {
        this.stageViewport.classList.remove('suspicious-active');
      }
    }

    // Anomaly Warning Banner
    if (this.anomalyBanner) {
      if (isSuspicious) {
        this.anomalyBanner.style.display = 'flex';
        if (this.anomalyFlagsEl) {
          const flags = scan.anomaly_flags && scan.anomaly_flags.length > 0 ? scan.anomaly_flags : ['ANOMALOUS_SEGMENTATION'];
          this.anomalyFlagsEl.innerHTML = flags.map(f => `<span class="scout-flag-pill">${f}</span>`).join('');
        }
      } else {
        this.anomalyBanner.style.display = 'none';
      }
    }

    // Header info
    if (this.viewerFilename) {
      this.viewerFilename.textContent = scan.filename;
      this.viewerFilename.title = scan.rel_path;
    }
    if (this.viewerBadge) {
      this.viewerBadge.className = `scout-card-badge ${isSuspicious ? 'critical' : 'clean'}`;
      this.viewerBadge.textContent = scan.severity;
    }

    // Set images for all 3 view modes
    // 1. Side-by-Side
    if (this.realImgSide) this.realImgSide.src = scan.image_url;
    if (this.maskImgSide) this.maskImgSide.src = scan.mask_url;

    // 2. Split Slider
    if (this.realImgSlider) this.realImgSlider.src = scan.image_url;
    if (this.maskImgSlider) this.maskImgSlider.src = scan.mask_url;

    // 3. Overlay Blend
    if (this.realImgOverlay) this.realImgOverlay.src = scan.image_url;
    if (this.maskImgOverlay) this.maskImgOverlay.src = scan.mask_url;

    // Fetch and render detailed diagnostic metrics
    this.updateMetrics(scan);
  }

  updateMetrics(scan) {
    const m = scan.metrics || {};
    if (this.mAreaRatio) {
      const pct = m.area_ratio != null ? `${(m.area_ratio * 100).toFixed(1)}%` : 'N/A';
      this.mAreaRatio.textContent = pct;
    }
    if (this.mThickness) {
      const mean = m.thickness_mean != null ? `${m.thickness_mean.toFixed(1)} px` : 'N/A';
      this.mThickness.textContent = mean;
    }
    if (this.mComponents) {
      const comps = m.foreground_components != null ? `${m.foreground_components}` : '1';
      this.mComponents.textContent = comps;
    }
    if (this.mDerivative) {
      const maxDy = m.max_dy_top != null ? `Δy ${Math.max(m.max_dy_top, m.max_dy_bot || 0).toFixed(1)}` : 'Pass';
      this.mDerivative.textContent = maxDy;
    }
    if (this.mMargin) {
      const margin = m.pitch_black_violations != null ? (m.pitch_black_violations > 0 ? 'FAIL' : 'PASS') : 'PASS';
      this.mMargin.textContent = margin;
      if (this.mMargin.parentElement) {
        if (margin === 'FAIL') this.mMargin.parentElement.classList.add('danger');
        else this.mMargin.parentElement.classList.remove('danger');
      }
    }
  }

  setToolType(toolType) {
    this.activeTool = toolType;
    this.updateToolUI();
  }

  toggleActiveTool(forceState = null) {
    if (forceState !== null) {
      this.isToolActive = forceState;
    } else {
      this.isToolActive = !this.isToolActive;
    }
    this.updateToolUI();
  }

  updateToolUI() {
    if (this.toolActionBtn) {
      if (this.isToolActive) {
        this.toolActionBtn.classList.add('active');
        if (this.activeTool === 'unet-window') {
          this.toolActionBtn.classList.add('unet-active');
        } else {
          this.toolActionBtn.classList.remove('unet-active');
        }
        if (this.toolActionText) {
          this.toolActionText.innerHTML = (this.activeTool === 'eraser')
            ? 'Eraser Active (Click Target)'
            : 'U-Net Active (Drag Window Box)';
        }
      } else {
        this.toolActionBtn.classList.remove('active', 'unet-active');
        if (this.toolActionText) {
          this.toolActionText.innerHTML = (this.activeTool === 'eraser')
            ? 'Surgical BG Eraser'
            : 'Rerun U-Net Window';
        }
      }
    }

    if (this.toolStatusEl) {
      this.toolStatusEl.style.display = this.isToolActive ? 'inline-block' : 'none';
      if (this.isToolActive) {
        this.toolStatusEl.textContent = (this.activeTool === 'eraser')
          ? 'Click on artifact / vitreous noise to surgically erase'
          : 'Drag a box window over tissue to rerun U-Net segmentation';
      }
    }

    const cursorClass = (this.activeTool === 'eraser') ? 'ablate-active' : 'unet-window-active';
    [this.canvasSideReal, this.canvasSideMask].forEach(canvas => {
      if (!canvas) return;
      canvas.classList.remove('ablate-active', 'unet-window-active');
      if (this.isToolActive) {
        canvas.classList.add(cursorClass);
      }
    });
  }

  setActionLoaderState(isLoading, text = '', progressPct = 0) {
    if (!this.toolActionBtn) return;

    if (isLoading) {
      this.toolActionBtn.classList.add('ablating-active');
      if (this.toolActionFill) {
        this.toolActionFill.style.width = `${progressPct}%`;
      }
      if (this.toolActionShimmer) {
        this.toolActionShimmer.style.display = 'block';
      }
      if (this.toolActionText) {
        this.toolActionText.innerHTML = `<span class="scout-btn-spinner"></span><span>${text}</span>`;
      }
    } else {
      this.toolActionBtn.classList.remove('ablating-active');
      if (this.toolActionFill) {
        this.toolActionFill.style.width = '0%';
      }
      if (this.toolActionShimmer) {
        this.toolActionShimmer.style.display = 'none';
      }
      if (this.toolActionText) {
        const defaultLabel = (this.activeTool === 'eraser')
          ? (this.isToolActive ? 'Eraser Active (Click Target)' : 'Surgical BG Eraser')
          : (this.isToolActive ? 'U-Net Active (Drag Window Box)' : 'Rerun U-Net Window');
        this.toolActionText.innerHTML = text || defaultLabel;
      }
    }
  }

  setUndoVisible(visible) {
    this.canUndo = visible;
    if (this.btnUndo) {
      this.btnUndo.style.display = visible ? 'inline-flex' : 'none';
    }
  }

  async executeAblation(x, y) {
    if (!this.selectedScan || this.isExecutingAction) return;
    this.isExecutingAction = true;

    // Start animated loader on the button
    this.setActionLoaderState(true, `Ablating [${x}, ${y}]...`, 30);

    let currentPct = 30;
    const progressTimer = setInterval(() => {
      if (currentPct < 90) {
        currentPct += 12;
        if (this.toolActionFill) {
          this.toolActionFill.style.width = `${currentPct}%`;
        }
      }
    }, 180);

    if (this.toolStatusEl) {
      this.toolStatusEl.style.display = 'inline-block';
      this.toolStatusEl.textContent = `Ablating noise at [${x}, ${y}]...`;
    }

    try {
      const res = await fetch('/api/scout/ablate_bg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rel_path: this.selectedScan.rel_path,
          x: x,
          y: y,
          radius_x: 0,
        }),
      });

      clearInterval(progressTimer);

      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        this.setActionLoaderState(false);
        alert(`Ablation failed: ${data.message || 'Unknown server error'}`);
        return;
      }

      if (this.toolActionFill) {
        this.toolActionFill.style.width = '100%';
      }
      if (this.toolActionText) {
        this.toolActionText.innerHTML = `Cleaned (${data.pixels_ablated} px)`;
      }

      this.applyActionResult(data, `Ablated ${data.pixels_ablated} noise px! Cleaned successfully.`);
    } catch (err) {
      clearInterval(progressTimer);
      this.setActionLoaderState(false);
      console.error('Error during surgical background ablation:', err);
      alert(`Network error during ablation: ${err.message}`);
    } finally {
      this.isExecutingAction = false;
    }
  }

  async executeUnetWindow(x1, y1, x2, y2) {
    if (!this.selectedScan || this.isExecutingAction) return;
    this.isExecutingAction = true;

    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2);
    const maxY = Math.max(y1, y2);

    this.setActionLoaderState(true, `Running U-Net on [${minX}, ${minY}, ${maxX}, ${maxY}]...`, 30);

    let currentPct = 30;
    const progressTimer = setInterval(() => {
      if (currentPct < 90) {
        currentPct += 14;
        if (this.toolActionFill) {
          this.toolActionFill.style.width = `${currentPct}%`;
        }
      }
    }, 180);

    if (this.toolStatusEl) {
      this.toolStatusEl.style.display = 'inline-block';
      this.toolStatusEl.textContent = `Running U-Net on window (${maxX - minX}x${maxY - minY} px)...`;
    }

    try {
      const res = await fetch('/api/scout/rerun_unet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rel_path: this.selectedScan.rel_path,
          x1: minX,
          y1: minY,
          x2: maxX,
          y2: maxY,
          threshold: 0.50,
        }),
      });

      clearInterval(progressTimer);

      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        this.setActionLoaderState(false);
        alert(`U-Net execution failed: ${data.message || 'Unknown server error'}`);
        return;
      }

      if (this.toolActionFill) {
        this.toolActionFill.style.width = '100%';
      }
      if (this.toolActionText) {
        this.toolActionText.innerHTML = `U-Net Refined (${data.pixels_changed} px)`;
      }

      this.applyActionResult(data, `U-Net window segmented (${data.pixels_changed} px updated)!`);
    } catch (err) {
      clearInterval(progressTimer);
      this.setActionLoaderState(false);
      console.error('Error during U-Net window execution:', err);
      alert(`Network error during U-Net execution: ${err.message}`);
    } finally {
      this.isExecutingAction = false;
    }
  }

  applyActionResult(data, successMessage) {
    this.selectedScan.severity = data.severity;
    this.selectedScan.is_suspicious = data.is_suspicious;
    this.selectedScan.anomaly_flags = data.anomaly_flags;
    this.selectedScan.metrics = data.metrics;

    const t = Date.now();
    const baseImgUrl = this.selectedScan.image_url.replace(/([&?])t=\d+/, '');
    const baseMaskUrl = this.selectedScan.mask_url.replace(/([&?])t=\d+/, '');
    const sepImg = baseImgUrl.includes('?') ? '&' : '?';
    const sepMask = baseMaskUrl.includes('?') ? '&' : '?';
    const updatedImgUrl = `${baseImgUrl}${sepImg}t=${t}`;
    const updatedMaskUrl = `${baseMaskUrl}${sepMask}t=${t}`;

    this.selectedScan.image_url = updatedImgUrl;
    this.selectedScan.mask_url = updatedMaskUrl;

    if (this.realImgSide) this.realImgSide.src = updatedImgUrl;
    if (this.maskImgSide) this.maskImgSide.src = updatedMaskUrl;
    if (this.realImgSlider) this.realImgSlider.src = updatedImgUrl;
    if (this.maskImgSlider) this.maskImgSlider.src = updatedMaskUrl;
    if (this.realImgOverlay) this.realImgOverlay.src = updatedImgUrl;
    if (this.maskImgOverlay) this.maskImgOverlay.src = updatedMaskUrl;

    const isSuspicious = data.is_suspicious || data.severity === 'WARNING' || data.severity === 'CRITICAL';
    if (this.stageViewport) {
      if (isSuspicious) this.stageViewport.classList.add('suspicious-active');
      else this.stageViewport.classList.remove('suspicious-active');
    }

    if (this.viewerBadge) {
      this.viewerBadge.className = `scout-card-badge ${isSuspicious ? 'critical' : 'clean'}`;
      this.viewerBadge.textContent = data.severity;
    }

    if (this.anomalyBanner) {
      if (isSuspicious) {
        this.anomalyBanner.style.display = 'flex';
        if (this.anomalyFlagsEl) {
          const flags = data.anomaly_flags && data.anomaly_flags.length > 0 ? data.anomaly_flags : ['ANOMALOUS_SEGMENTATION'];
          this.anomalyFlagsEl.innerHTML = flags.map(f => `<span class="scout-flag-pill">${f}</span>`).join('');
        }
      } else {
        this.anomalyBanner.style.display = 'none';
      }
    }

    this.updateMetrics(data);
    this.setUndoVisible(true);

    if (this.toolStatusEl) {
      this.toolStatusEl.textContent = successMessage;
      setTimeout(() => {
        if (this.isToolActive) {
          this.toolStatusEl.textContent = (this.activeTool === 'eraser')
            ? 'Click on artifact / vitreous noise to surgically erase'
            : 'Drag a box window over tissue to rerun U-Net segmentation';
        }
      }, 3000);
    }

    const cardEl = this.listEl ? this.listEl.querySelector(`.scout-card[data-index="${this.selectedIndex}"]`) : null;
    if (cardEl) {
      const thumb = cardEl.querySelector('.scout-card-thumb');
      if (thumb) thumb.src = updatedMaskUrl;
      const badge = cardEl.querySelector('.scout-card-badge');
      if (badge) {
        badge.className = `scout-card-badge ${isSuspicious ? 'critical' : 'clean'}`;
        badge.textContent = data.severity;
      }
      if (isSuspicious) cardEl.classList.add('suspicious');
      else cardEl.classList.remove('suspicious');
    }

    this.loadOverview();

    setTimeout(() => {
      this.setActionLoaderState(false);
    }, 1200);
  }

  async undoLastAction() {
    if (!this.selectedScan || this.isExecutingAction) return;
    this.isExecutingAction = true;

    if (this.toolStatusEl) {
      this.toolStatusEl.style.display = 'inline-block';
      this.toolStatusEl.textContent = 'Restoring previous mask state...';
    }

    try {
      const res = await fetch('/api/scout/undo_ablate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rel_path: this.selectedScan.rel_path,
        }),
      });

      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        alert(`Undo failed: ${data.message || 'Unknown server error'}`);
        return;
      }

      this.selectedScan.severity = data.severity;
      this.selectedScan.is_suspicious = data.is_suspicious;
      this.selectedScan.anomaly_flags = data.anomaly_flags;
      this.selectedScan.metrics = data.metrics;

      const t = Date.now();
      const baseImgUrl = this.selectedScan.image_url.replace(/([&?])t=\d+/, '');
      const baseMaskUrl = this.selectedScan.mask_url.replace(/([&?])t=\d+/, '');
      const sepImg = baseImgUrl.includes('?') ? '&' : '?';
      const sepMask = baseMaskUrl.includes('?') ? '&' : '?';
      const updatedImgUrl = `${baseImgUrl}${sepImg}t=${t}`;
      const updatedMaskUrl = `${baseMaskUrl}${sepMask}t=${t}`;

      this.selectedScan.image_url = updatedImgUrl;
      this.selectedScan.mask_url = updatedMaskUrl;

      if (this.realImgSide) this.realImgSide.src = updatedImgUrl;
      if (this.maskImgSide) this.maskImgSide.src = updatedMaskUrl;
      if (this.realImgSlider) this.realImgSlider.src = updatedImgUrl;
      if (this.maskImgSlider) this.maskImgSlider.src = updatedMaskUrl;
      if (this.realImgOverlay) this.realImgOverlay.src = updatedImgUrl;
      if (this.maskImgOverlay) this.maskImgOverlay.src = updatedMaskUrl;

      const isSuspicious = data.is_suspicious || data.severity === 'WARNING' || data.severity === 'CRITICAL';
      if (this.stageViewport) {
        if (isSuspicious) this.stageViewport.classList.add('suspicious-active');
        else this.stageViewport.classList.remove('suspicious-active');
      }

      if (this.viewerBadge) {
        this.viewerBadge.className = `scout-card-badge ${isSuspicious ? 'critical' : 'clean'}`;
        this.viewerBadge.textContent = data.severity;
      }

      if (this.anomalyBanner) {
        if (isSuspicious) {
          this.anomalyBanner.style.display = 'flex';
          if (this.anomalyFlagsEl) {
            const flags = data.anomaly_flags && data.anomaly_flags.length > 0 ? data.anomaly_flags : ['ANOMALOUS_SEGMENTATION'];
            this.anomalyFlagsEl.innerHTML = flags.map(f => `<span class="scout-flag-pill">${f}</span>`).join('');
          }
        } else {
          this.anomalyBanner.style.display = 'none';
        }
      }

      this.updateMetrics(data);
      this.setUndoVisible(false);

      if (this.toolStatusEl) {
        this.toolStatusEl.textContent = 'Reverted to backup state.';
        setTimeout(() => {
          if (this.isToolActive) {
            this.toolStatusEl.textContent = (this.activeTool === 'eraser')
              ? 'Click on artifact / vitreous noise to surgically erase'
              : 'Drag a box window over tissue to rerun U-Net segmentation';
          } else {
            this.toolStatusEl.style.display = 'none';
          }
        }, 3000);
      }

      const cardEl = this.listEl ? this.listEl.querySelector(`.scout-card[data-index="${this.selectedIndex}"]`) : null;
      if (cardEl) {
        const thumb = cardEl.querySelector('.scout-card-thumb');
        if (thumb) thumb.src = updatedMaskUrl;
        const badge = cardEl.querySelector('.scout-card-badge');
        if (badge) {
          badge.className = `scout-card-badge ${isSuspicious ? 'critical' : 'clean'}`;
          badge.textContent = data.severity;
        }
        if (isSuspicious) cardEl.classList.add('suspicious');
        else cardEl.classList.remove('suspicious');
      }

      this.loadOverview();
    } catch (err) {
      console.error('Error during undo action:', err);
      alert(`Network error during undo: ${err.message}`);
    } finally {
      this.isExecutingAction = false;
    }
  }

  clearViewer() {
    this.selectedIndex = -1;
    this.selectedScan = null;
    this.setUndoVisible(false);
    this.setActionLoaderState(false);
    this.toggleActiveTool(false);
    if (this.stageViewport) this.stageViewport.classList.remove('suspicious-active');
    if (this.anomalyBanner) this.anomalyBanner.style.display = 'none';
    if (this.viewerFilename) this.viewerFilename.textContent = 'No scan selected';
    if (this.realImgSide) this.realImgSide.src = '';
    if (this.maskImgSide) this.maskImgSide.src = '';
  }
}

// Attach globally
window.ScoutExplorer = ScoutExplorer;
