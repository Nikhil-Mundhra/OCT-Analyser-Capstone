/**
 * Network API Client
 */

async function fetchFolders() {
  const res = await fetch('/api/folders');
  if (!res.ok) {
    throw new Error(`Failed to fetch folders: ${res.statusText}`);
  }
  return await res.json();
}

async function reprocessFolder(folder, params, isRandomRefresh = false) {
  const res = await fetch('/api/reprocess', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      params,
      random_sample: isRandomRefresh
    })
  });
  if (!res.ok) {
    throw new Error(`Failed to reprocess folder: ${res.statusText}`);
  }
  return await res.json();
}

async function reprocessSingleImage(folder, filename, params) {
  const res = await fetch('/api/reprocess_single', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      filename,
      params
    })
  });
  if (!res.ok) {
    throw new Error(`Failed to reprocess single image: ${res.statusText}`);
  }
  return await res.json();
}

async function fetchCuratedManifest() {
  const res = await fetch('/api/curated_manifest');
  if (!res.ok) {
    throw new Error(`Failed to fetch curated manifest: ${res.statusText}`);
  }
  return await res.json();
}

async function curateSample(folder, filename, params) {
  const res = await fetch('/api/curate_sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      filename,
      params
    })
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to curate sample: ${res.statusText}`);
  }
  return await res.json();
}

async function uncurateSample(folder, filename) {
  const res = await fetch('/api/uncurate_sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      filename
    })
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to uncurate sample: ${res.statusText}`);
  }
  return await res.json();
}

async function curateBatch(folder, filenames, params) {
  const res = await fetch('/api/curate_batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      filenames,
      params
    })
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to curate batch: ${res.statusText}`);
  }
  return await res.json();
}

async function fetchCropFilterQueue(folder = null, offset = 0, limit = 30, clearCache = false) {
  let url = `/api/crop_filter_queue?offset=${offset}&limit=${limit}`;
  if (folder && folder !== 'ALL') {
    url += `&folder=${encodeURIComponent(folder)}`;
  }
  if (clearCache) {
    url += `&clear_cache=true`;
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch crop filter queue: ${res.statusText}`);
  }
  return await res.json();
}

async function saveCuratedCrop(
  folder,
  filename,
  y_top_points,
  y_bot_points,
  crop_left = 0,
  crop_right = null,
  crop_left_top = null,
  crop_left_bot = null,
  crop_right_top = null,
  crop_right_bot = null
) {
  const res = await fetch('/api/save_curated_crop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder,
      filename,
      y_top_points,
      y_bot_points,
      crop_left,
      crop_right,
      crop_left_top,
      crop_left_bot,
      crop_right_top,
      crop_right_bot
    })
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to save curated crop: ${res.statusText}`);
  }
  return await res.json();
}

async function fetchCurationStats() {
  const res = await fetch('/api/curation_stats');
  if (!res.ok) {
    throw new Error(`Failed to fetch curation stats: ${res.statusText}`);
  }
  return await res.json();
}

async function triggerUNetRetrain(epochs = 10, lr = 2e-4) {
  const res = await fetch('/api/retrain_unet', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ epochs, lr })
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to trigger retraining: ${res.statusText}`);
  }
  return await res.json();
}

async function fetchRetrainStatus() {
  const res = await fetch('/api/retrain_status');
  if (!res.ok) {
    throw new Error(`Failed to fetch retrain status: ${res.statusText}`);
  }
  return await res.json();
}

async function stopUNetRetraining() {
  const res = await fetch('/api/stop_retrain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to stop retraining: ${res.statusText}`);
  }
  return await res.json();
}

async function fetchTrainingHistory() {
  const res = await fetch('/api/training_history');
  if (!res.ok) {
    throw new Error(`Failed to fetch training history: ${res.statusText}`);
  }
  return await res.json();
}
