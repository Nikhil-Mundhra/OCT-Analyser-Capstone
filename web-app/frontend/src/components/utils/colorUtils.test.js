import { describe, it, expect } from 'vitest';
import { getClassColor } from './colorUtils';

describe('getClassColor', () => {
  it('should return correct color for IRF', () => {
    const color = getClassColor('IRF');
    expect(color.fill).toBe('rgba(255,255,255,0.3)');
    expect(color.stroke).toBe('rgba(255,255,255,0.8)');
  });

  it('should return correct color for SRF', () => {
    const color = getClassColor('SRF');
    expect(color.fill).toBe('rgba(239,68,68,0.3)');
    expect(color.stroke).toBe('rgba(239,68,68,0.8)');
  });

  it('should return default color for unknown classes', () => {
    const color = getClassColor('UNKNOWN_CLASS');
    expect(color.fill).toBe('rgba(34,197,94,0.3)');
    expect(color.stroke).toBe('rgba(34,197,94,0.8)');
  });
});
