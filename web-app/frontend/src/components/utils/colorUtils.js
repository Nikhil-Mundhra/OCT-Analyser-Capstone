/**
 * Segmentation class colour palette.
 * Returns a structured { fill, stroke } object — avoids the fragile
 * rgba-string-replace pattern that was used before.
 *
 * @param {string} className  - segmentation class name (e.g. "IRF", "SRF")
 * @returns {{ fill: string, stroke: string }}
 */
const PALETTE = {
  IRF: { r: 255, g: 255, b: 255 }, // White  — Intraretinal Fluid
  SRF: { r: 239, g: 68,  b: 68  }, // Red    — Subretinal Fluid
};

const DEFAULT_COLOR = { r: 34, g: 197, b: 94 }; // Green — normal tissue

export function getClassColor(className) {
  const { r, g, b } = PALETTE[className] ?? DEFAULT_COLOR;
  return {
    fill:   `rgba(${r},${g},${b},0.3)`,
    stroke: `rgba(${r},${g},${b},0.8)`,
  };
}
