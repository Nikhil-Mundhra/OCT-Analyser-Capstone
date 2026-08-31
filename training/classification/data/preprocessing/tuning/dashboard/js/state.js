/**
 * Application State & Parameter Schema
 */

const PARAM_SCHEMA = [
  { key: 'auto_mode', type: 'bool', default: true },
  { key: 'top_noise_mult', type: 'float', default: 1.5, min: 0.5, max: 8.0, step: 0.1 },
  { key: 'use_dp_ilm', type: 'bool', default: true },
  { key: 'ilm_gradient_weight', type: 'float', default: 0.70, min: 0.3, max: 1.0, step: 0.05 },
  { key: 'ilm_smooth_weight', type: 'float', default: 0.25, min: 0.05, max: 1.0, step: 0.05 },
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
  { key: 'sfcm_slack_bottom_px', type: 'int', default: 20, min: 0, max: 60, step: 2 },
  { key: 'sfcm_gaussian_sigma', type: 'int', default: 15, min: 1, max: 40, step: 1 },
  { key: 'sfcm_n_clusters', type: 'int', default: 3, min: 2, max: 5, step: 1 },
  { key: 'sfcm_fuzziness_m', type: 'float', default: 2.0, min: 1.1, max: 3.5, step: 0.1 },
  { key: 'rpe_smooth_weight', type: 'float', default: 0.20, min: 0.02, max: 1.0, step: 0.02 },
  { key: 'rpe_depth_weight', type: 'float', default: 0.40, min: 0.0, max: 1.0, step: 0.05 },
  { key: 'rpe_gradient_weight', type: 'float', default: 0.30, min: 0.0, max: 1.0, step: 0.05 },
  { key: 'rpe_bottom_env_size', type: 'int', default: 15, min: 1, max: 35, step: 2 },
  { key: 'detect_caverns', type: 'bool', default: false },
  { key: 'holes_enabled', type: 'bool', default: true },
  { key: 'hole_min_area', type: 'int', default: 25, min: 5, max: 200, step: 5 },
  { key: 'hole_max_area', type: 'int', default: 15000, min: 500, max: 25000, step: 500 },
  { key: 'hole_contrast_offset', type: 'int', default: 8, min: 2, max: 25, step: 1 },
  { key: 'hole_local_window', type: 'int', default: 15, min: 5, max: 61, step: 2 },
  { key: 'hole_max_aspect_ratio', type: 'float', default: 2.8, min: 1.2, max: 5.0, step: 0.1 },
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
