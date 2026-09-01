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

