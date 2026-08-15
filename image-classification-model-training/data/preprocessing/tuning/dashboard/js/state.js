/**
 * Application State & Parameter Schema
 */

const PARAM_SCHEMA = [
  { key: 'top_noise_mult', type: 'float', default: 1.5, min: 0.5, max: 8.0, step: 0.1 },
  { key: 'bot_noise_mult', type: 'float', default: 3.0, min: 1.5, max: 5.0, step: 0.1 },
  { key: 'use_otsu_bottom', type: 'bool', default: true },
  { key: 'use_sfcm', type: 'bool', default: false },
  { key: 'shadow_bridge_top_pct', type: 'int', default: 20, min: 5, max: 40, step: 1 },
  { key: 'shadow_bridge_bot_pct', type: 'int', default: 20, min: 5, max: 40, step: 1 },
  { key: 'gaussian_sigma', type: 'int', default: 15, min: 1, max: 40, step: 1 },
  { key: 'margin_top', type: 'int', default: 15, min: 5, max: 30, step: 1 },
  { key: 'margin_bottom', type: 'int', default: 15, min: 0, max: 80, step: 2 },
  { key: 'top_spike_suppress_px', type: 'int', default: 0, min: 0, max: 120, step: 2 },
  { key: 'top_spike_window_px', type: 'int', default: 80, min: 10, max: 200, step: 5 },
  { key: 'top_dip_suppress_px', type: 'int', default: 0, min: 0, max: 120, step: 2 },
  { key: 'top_dip_window_px', type: 'int', default: 80, min: 10, max: 200, step: 5 },
  { key: 'sfcm_margin_bottom', type: 'int', default: 15, min: 0, max: 80, step: 2 },
  { key: 'sfcm_gaussian_sigma', type: 'int', default: 15, min: 1, max: 40, step: 1 },
  { key: 'sfcm_n_clusters', type: 'int', default: 3, min: 2, max: 5, step: 1 },
  { key: 'sfcm_fuzziness_m', type: 'float', default: 2.0, min: 1.1, max: 3.5, step: 0.1 },
  { key: 'compass_ui_enabled', type: 'bool', default: false },
  { key: 'compass_location', type: 'str', default: 'auto' }
];

let currentData = {
  folders: [],
  saved_params: {},
  default_params: {}
};

let currentFolder = '';
let debounceTimer = null;
let drawVectorsEnabled = true;
let activeHandle = null;
let draggingHandle = null;
