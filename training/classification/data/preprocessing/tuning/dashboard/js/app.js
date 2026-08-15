/**
 * Main Application Lifecycle, DOM Rendering, and Event Listeners
 */

function getParamEl(key) {
  return document.getElementById(`param-${key}`);
}

function getValEl(key) {
  return document.getElementById(`val-${key}`);
}

function updateSliderLabel(key, text) {
  const el = getValEl(key);
  if (el) el.textContent = text;
}

function updateOtsuSfcmVisibility() {
  const isSfcmOn = getParamEl('use_sfcm') ? getParamEl('use_sfcm').checked : false;
  const isOtsuOn = getParamEl('use_otsu_bottom') ? getParamEl('use_otsu_bottom').checked : true;

  const otsuGroup = document.getElementById('group-otsu-bottom-params');
  const sfcmGroup = document.getElementById('group-sfcm-params');

  if (otsuGroup) {
    otsuGroup.style.display = isOtsuOn ? 'flex' : 'none';
  }
  if (sfcmGroup) {
    sfcmGroup.style.display = isSfcmOn ? 'flex' : 'none';
  }
}

function loadFolderParams(folder) {
  currentFolder = folder;
  const params = currentData.saved_params[folder] || currentData.default_params;

  PARAM_SCHEMA.forEach(field => {
    const el = getParamEl(field.key);
    if (!el) return;

    let val = params[field.key] !== undefined ? params[field.key] : field.default;

    if (field.type === 'bool') {
      el.checked = Boolean(val);
    } else if (field.type === 'str') {
      el.value = String(val);
    } else {
      el.value = val;
      const unit = (field.key.includes('pct') ? '%' : (field.key.includes('margin') || field.key.includes('px') ? 'px' : ''));
      updateSliderLabel(field.key, val + unit);
    }
  });

  updateOtsuSfcmVisibility();
  updateJsonEditorFromUI();
}

function getParamsFromUI() {
  const params = {};
  PARAM_SCHEMA.forEach(field => {
    const el = getParamEl(field.key);
    if (!el) return;

    if (field.type === 'bool') {
      params[field.key] = el.checked;
    } else if (field.type === 'int') {
      params[field.key] = parseInt(el.value, 10);
    } else if (field.type === 'float') {
      params[field.key] = parseFloat(el.value);
    } else {
      params[field.key] = el.value;
    }
  });
  return params;
}

function updateJsonEditorFromUI() {
  const editor = document.getElementById('json-editor');
  if (!editor) return;
  const currentParams = getParamsFromUI();
  editor.value = JSON.stringify(currentParams, null, 2);
  const errorEl = document.getElementById('json-error');
  if (errorEl) errorEl.classList.remove('active');
}

function applyJsonToUI() {
  const editor = document.getElementById('json-editor');
  const errorEl = document.getElementById('json-error');
  try {
    const parsed = JSON.parse(editor.value);
    errorEl.classList.remove('active');

    PARAM_SCHEMA.forEach(field => {
      if (parsed[field.key] !== undefined) {
        const el = getParamEl(field.key);
        if (!el) return;
        if (field.type === 'bool') {
          el.checked = Boolean(parsed[field.key]);
        } else if (field.type === 'str') {
          el.value = String(parsed[field.key]);
        } else {
          el.value = parsed[field.key];
          const unit = (field.key.includes('pct') ? '%' : (field.key.includes('margin') || field.key.includes('px') ? 'px' : ''));
          updateSliderLabel(field.key, parsed[field.key] + unit);
        }
      }
    });

    updateOtsuSfcmVisibility();
    triggerReprocess(false, false);
    switchPanel('slider');
  } catch (err) {
    errorEl.textContent = 'JSON Syntax Error: ' + err.message;
    errorEl.classList.add('active');
  }
}

function switchPanel(mode) {
  const pSlider = document.getElementById('panel-sliders');
  const pJson = document.getElementById('panel-json');
  const btnSlider = document.getElementById('btn-mode-slider');
  const btnJson = document.getElementById('btn-mode-json');

  if (mode === 'slider') {
    pSlider.style.display = 'flex';
    pJson.style.display = 'none';
    btnSlider.classList.add('active');
    btnJson.classList.remove('active');
  } else {
    updateJsonEditorFromUI();
    pSlider.style.display = 'none';
    pJson.style.display = 'flex';
    btnSlider.classList.remove('active');
    btnJson.classList.add('active');
  }
}

function scheduleDebouncedReprocess() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    updateJsonEditorFromUI();
    triggerReprocess(false, false);
  }, 250);
}

function showStatusMessage(text, durationMs = 2500) {
  const statusEl = document.getElementById('status-msg');
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.classList.add('active');
  setTimeout(() => statusEl.classList.remove('active'), durationMs);
}

async function triggerReprocess(isSaveNotification = false, isRandomRefresh = false) {
  const folder = document.getElementById('folder-select').value;
  const params = getParamsFromUI();

  try {
    const data = await reprocessFolder(folder, params, isRandomRefresh);
    renderGallery(data.samples);

    if (isSaveNotification) {
      showStatusMessage('Saved all folder parameters to folder_params.json');
    } else if (isRandomRefresh) {
      showStatusMessage('Fetched 6 New Random Sample Images');
    } else {
      showStatusMessage('Live Parameters Updated');
    }
  } catch (err) {
    showStatusMessage('Error: ' + err.message, 4000);
  }
}

function copyImageConfig(btn) {
  const filename = btn.getAttribute('data-filename');
  const filepath = btn.getAttribute('data-filepath');
  const params = getParamsFromUI();
  const folder = document.getElementById('folder-select').value;

  const locText = filepath ? filepath : `${folder}/${filename}`;
  const textToCopy = `Image Location: ${locText}\nJSON Config:\n${JSON.stringify(params, null, 2)}`;

  navigator.clipboard.writeText(textToCopy).then(() => {
    const origText = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.borderColor = 'var(--accent-green)';
    btn.style.color = 'var(--accent-green)';
    setTimeout(() => {
      btn.textContent = origText;
      btn.style.borderColor = '';
      btn.style.color = '';
    }, 1500);
  });
}

function renderGallery(samples) {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '';

  if (!samples || samples.length === 0) {
    grid.innerHTML = '<div style="color: var(--text-muted); grid-column: 1/-1; padding: 40px; text-align: center;">No valid images found in folder.</div>';
    return;
  }

  const isOtsuBottomActive = getParamEl('use_otsu_bottom') ? getParamEl('use_otsu_bottom').checked : true;
  const isSfcmActive = getParamEl('use_sfcm') ? getParamEl('use_sfcm').checked : false;

  samples.forEach(s => {
    const card = document.createElement('div');
    card.className = 'image-card';

    const topPathD = buildSvgPathD(s.top_vector);
    const botPathD = isOtsuBottomActive ? buildSvgPathD(s.bottom_vector) : '';

    const hasSfcm = isSfcmActive && s.sfcm_vector && s.sfcm_vector.length > 0;
    const sfcmPathD = hasSfcm ? buildSvgPathD(s.sfcm_vector) : '';
    const sfcmStartVec = (hasSfcm && s.rpe_vector && s.rpe_vector.length > 0) ? s.rpe_vector : s.top_vector;
    const sfcmMaskD = hasSfcm ? buildSfcmMaskPolygonD(sfcmStartVec, s.sfcm_vector) : '';

    const topHandlesHtml = buildSvgHandlesHtml(s.top_vector, 'top');
    const botHandlesHtml = isOtsuBottomActive ? buildSvgHandlesHtml(s.bottom_vector, 'bottom') : '';
    const sfcmHandlesHtml = hasSfcm ? buildSvgHandlesHtml(s.sfcm_vector, 'bottom', [6, 19, 32, 45, 57], true) : '';

    const imgScale = s.scale || 0.437;
    const padT = s.pad_t || 0;

    let tagText = 'Otsu (Top Only)';
    if (isOtsuBottomActive && hasSfcm) {
      tagText = 'Otsu + SFCM Choroid (Orange)';
    } else if (hasSfcm) {
      tagText = 'Otsu Top + SFCM Choroid (Orange)';
    } else if (isOtsuBottomActive) {
      tagText = 'Otsu (Top & Bottom)';
    }

    const safeId = s.filename.replace(/[^a-zA-Z0-9]/g, '-');
    card.innerHTML = `
      <div class="card-header-bar">
        <span>${s.filename}</span>
        <div style="display: flex; align-items: center; gap: 6px;">
          <button class="btn-refresh-single" data-filename="${s.filename}" data-filepath="${s.filepath || ''}" onclick="refreshSingleCard(this)">Refresh Image</button>
          <button class="btn-copy-config" data-filename="${s.filename}" data-filepath="${s.filepath || ''}" onclick="copyImageConfig(this)">Copy Config</button>
        </div>
      </div>
      <div class="image-comparison-row">
        <div class="img-wrap">
          <img src="${s.raw_url}?t=${Date.now()}" alt="Raw">
          <span class="img-tag">Raw Classified Scan</span>
        </div>
        <div class="img-wrap" id="wrap-${safeId}">
          <img src="${s.proc_url}?t=${Date.now()}" alt="Preprocessed">
          <svg class="vector-svg-overlay" viewBox="0 0 384 384"
               data-img-scale="${imgScale}" data-pad-t="${padT}"
               style="display: ${drawVectorsEnabled ? 'block' : 'none'}; user-select:none;">
            ${hasSfcm ? `<path d="${sfcmMaskD}" fill="rgba(255, 145, 0, 0.35)" stroke="none" style="pointer-events:none;"/>` : ''}
            <path d="${topPathD}" stroke="#00f2fe" stroke-width="2" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 3px #00f2fe)" style="pointer-events:none;"/>
            ${isOtsuBottomActive ? `<path d="${botPathD}" stroke="#ff007f" stroke-width="2" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 3px #ff007f)" style="pointer-events:none;"/>` : ''}
            ${hasSfcm ? `<path d="${sfcmPathD}" stroke="#ff9100" stroke-width="2.5" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 4px #ff9100)" style="pointer-events:none;"/>` : ''}
            ${topHandlesHtml}
            ${botHandlesHtml}
            ${sfcmHandlesHtml}
          </svg>
          <span class="img-tag">${tagText}</span>
        </div>
      </div>
    `;

    const cardSvg = card.querySelector('.vector-svg-overlay');
    card.querySelectorAll('.top-handle').forEach(circle => {
      circle.addEventListener('mousedown', e => {
        onHandleDragStart(e, cardSvg, 'top', parseFloat(circle.dataset.handleY));
      });
    });
    card.querySelectorAll('.bot-handle').forEach(circle => {
      circle.addEventListener('mousedown', e => {
        onHandleDragStart(e, cardSvg, 'bottom', parseFloat(circle.dataset.handleY));
      });
    });

    grid.appendChild(card);
  });
}

async function refreshSingleCard(btn) {
  const filename = btn.getAttribute('data-filename');
  const folderName = document.getElementById('folder-select').value;
  const params = getParamsFromUI();

  btn.textContent = 'Processing...';
  btn.style.opacity = '0.6';

  try {
    const data = await reprocessSingleImage(folderName, filename, params);
    if (data.status === 'success' && data.sample) {
      const s = data.sample;
      const safeId = s.filename.replace(/[^a-zA-Z0-9]/g, '-');
      const wrap = document.getElementById(`wrap-${safeId}`);
      if (wrap) {
        const isOtsuBottomActive = getParamEl('use_otsu_bottom') ? getParamEl('use_otsu_bottom').checked : true;
        const isSfcmActive = getParamEl('use_sfcm') ? getParamEl('use_sfcm').checked : false;

        const topPathD = buildSvgPathD(s.top_vector);
        const botPathD = isOtsuBottomActive ? buildSvgPathD(s.bottom_vector) : '';

        const hasSfcm = isSfcmActive && s.sfcm_vector && s.sfcm_vector.length > 0;
        const sfcmPathD = hasSfcm ? buildSvgPathD(s.sfcm_vector) : '';
        const sfcmStartVec = (hasSfcm && s.rpe_vector && s.rpe_vector.length > 0) ? s.rpe_vector : s.top_vector;
        const sfcmMaskD = hasSfcm ? buildSfcmMaskPolygonD(sfcmStartVec, s.sfcm_vector) : '';

        const topHandlesHtml = buildSvgHandlesHtml(s.top_vector, 'top');
        const botHandlesHtml = isOtsuBottomActive ? buildSvgHandlesHtml(s.bottom_vector, 'bottom') : '';
        const sfcmHandlesHtml = hasSfcm ? buildSvgHandlesHtml(s.sfcm_vector, 'bottom', [6, 19, 32, 45, 57], true) : '';

        const imgScale = s.scale || 0.437;
        const padT = s.pad_t || 0;

        let tagText = 'Otsu (Top Only)';
        if (isOtsuBottomActive && hasSfcm) {
          tagText = 'Otsu + SFCM Choroid (Orange)';
        } else if (hasSfcm) {
          tagText = 'Otsu Top + SFCM Choroid (Orange)';
        } else if (isOtsuBottomActive) {
          tagText = 'Otsu (Top & Bottom)';
        }

        wrap.innerHTML = `
          <img src="${s.proc_url}?t=${Date.now()}" alt="Preprocessed">
          <svg class="vector-svg-overlay" viewBox="0 0 384 384"
               data-img-scale="${imgScale}" data-pad-t="${padT}"
               style="display: ${drawVectorsEnabled ? 'block' : 'none'}; user-select:none;">
            ${hasSfcm ? `<path d="${sfcmMaskD}" fill="rgba(255, 145, 0, 0.35)" stroke="none" style="pointer-events:none;"/>` : ''}
            <path d="${topPathD}" stroke="#00f2fe" stroke-width="2" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 3px #00f2fe)" style="pointer-events:none;"/>
            ${isOtsuBottomActive ? `<path d="${botPathD}" stroke="#ff007f" stroke-width="2" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 3px #ff007f)" style="pointer-events:none;"/>` : ''}
            ${hasSfcm ? `<path d="${sfcmPathD}" stroke="#ff9100" stroke-width="2.5" fill="none" stroke-linecap="round" filter="drop-shadow(0 0 4px #ff9100)" style="pointer-events:none;"/>` : ''}
            ${topHandlesHtml}
            ${botHandlesHtml}
            ${sfcmHandlesHtml}
          </svg>
          <span class="img-tag">${tagText}</span>
        `;

        const cardSvg = wrap.querySelector('.vector-svg-overlay');
        wrap.querySelectorAll('.top-handle').forEach(circle => {
          circle.addEventListener('mousedown', e => {
            onHandleDragStart(e, cardSvg, 'top', parseFloat(circle.dataset.handleY));
          });
        });
        wrap.querySelectorAll('.bot-handle').forEach(circle => {
          circle.addEventListener('mousedown', e => {
            onHandleDragStart(e, cardSvg, 'bottom', parseFloat(circle.dataset.handleY));
          });
        });

        showStatusMessage(`Updated ${s.filename}`);
      }
    }
  } catch (err) {
    showStatusMessage(`Failed to update ${filename}: ${err.message}`, 4000);
  } finally {
    btn.textContent = 'Refresh Image';
    btn.style.opacity = '1';
  }
}

function initParamGroups() {
  document.querySelectorAll('.param-group').forEach(group => {
    group.addEventListener('toggle', () => {
      if (group.open) {
        group.setAttribute('data-pinned', 'true');
      } else {
        group.removeAttribute('data-pinned');
      }
    });
  });
}

async function init() {
  initParamGroups();

  try {
    currentData = await fetchFolders();

    const folderSelect = document.getElementById('folder-select');
    folderSelect.innerHTML = '';

    currentData.folders.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      folderSelect.appendChild(opt);
    });

    folderSelect.addEventListener('change', e => {
      loadFolderParams(e.target.value);
      triggerReprocess(false, false);
    });

    PARAM_SCHEMA.forEach(field => {
      const el = getParamEl(field.key);
      if (!el) return;

      const eventName = (field.type === 'bool' || field.type === 'str') ? 'change' : 'input';
      el.addEventListener(eventName, e => {
        if (field.type === 'bool') {
          updateOtsuSfcmVisibility();
        } else if (field.type !== 'str') {
          const unit = (field.key.includes('pct') ? '%' : (field.key.includes('margin') || field.key.includes('px') ? 'px' : ''));
          updateSliderLabel(field.key, e.target.value + unit);
        }
        scheduleDebouncedReprocess();
      });
    });

    document.getElementById('btn-save').addEventListener('click', () => {
      triggerReprocess(true, false);
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
      const folder = document.getElementById('folder-select').value;
      const def = currentData.default_params;
      currentData.saved_params[folder] = JSON.parse(JSON.stringify(def));
      loadFolderParams(folder);
      triggerReprocess(false, false);
      showStatusMessage('Reset to Default Parameters');
    });

    document.getElementById('btn-refresh-samples').addEventListener('click', () => {
      triggerReprocess(false, true);
    });

    document.getElementById('btn-toggle-vectors').addEventListener('click', e => {
      drawVectorsEnabled = !drawVectorsEnabled;
      e.target.textContent = drawVectorsEnabled ? 'Hide Boundary Lines' : 'Show Boundary Lines';
      document.querySelectorAll('.vector-svg-overlay').forEach(svg => {
        svg.style.display = drawVectorsEnabled ? 'block' : 'none';
      });
    });

    document.getElementById('btn-mode-slider').addEventListener('click', () => switchPanel('slider'));
    document.getElementById('btn-mode-json').addEventListener('click', () => switchPanel('json'));
    document.getElementById('btn-apply-json').addEventListener('click', applyJsonToUI);

    const targetFolder = currentData.folders[0] || '';
    if (targetFolder) {
      loadFolderParams(targetFolder);
      triggerReprocess(false, false);
    }
  } catch (err) {
    showStatusMessage('Initialization failed: ' + err.message, 5000);
  }
}

window.addEventListener('DOMContentLoaded', init);
