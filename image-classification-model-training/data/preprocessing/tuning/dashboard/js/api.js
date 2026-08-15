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
