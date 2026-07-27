import { describe, it, expect } from 'vitest';
import { normalizeScanResult } from './octAnalyzerClient';

describe('normalizeScanResult', () => {
  it('should return the original input if it is not an object', () => {
    expect(normalizeScanResult(null)).toBeNull();
    expect(normalizeScanResult(undefined)).toBeUndefined();
    expect(normalizeScanResult('string')).toBe('string');
  });

  it('should normalize a basic scan object', () => {
    const scan = {
      scan_id: '123',
      status: 'completed',
      diagnosis: 'AMD_WET',
      confidence: 0.95
    };
    
    const result = normalizeScanResult(scan);
    
    expect(result.scan_id).toBe('123');
    expect(result.status).toBe('completed');
    expect(result.diagnosis).toBe('AMD_WET');
    expect(result.confidence).toBe(0.95);
    expect(result.level1).toEqual({});
    expect(result.level2).toEqual({});
    expect(result.level3).toEqual({});
    expect(result.gradcams).toEqual({});
    expect(result.previews).toEqual({});
    expect(result.segmentation).toBeNull();
  });

  it('should handle nested classification objects', () => {
    const scan = {
      id: 'abc',
      classification: {
        diagnosis: 'NORMAL',
        confidence: 0.99,
        level1: { prediction: 'NORMAL' },
        level2: {},
        level3: {}
      }
    };
    
    const result = normalizeScanResult(scan);
    
    expect(result.scan_id).toBe('abc');
    expect(result.diagnosis).toBe('NORMAL');
    expect(result.confidence).toBe(0.99);
    expect(result.level1).toEqual({ prediction: 'NORMAL' });
  });

  it('should attach segmentation if provided', () => {
    const scan = { id: '123' };
    const segmentation = { layers: [], lesions: [] };
    
    const result = normalizeScanResult(scan, segmentation);
    
    expect(result.segmentation).toEqual(segmentation);
  });

  it('should preserve data: base64 URLs in preview map without prepending slash', () => {
    const scan = {
      id: '123',
      previews: {
        raw: 'data:image/jpeg;base64,/9j/4AAQ...',
        unet_overlay: 'data:image/jpeg;base64,/9j/4AAQ...'
      }
    };

    const result = normalizeScanResult(scan);

    expect(result.previews.raw).toBe('data:image/jpeg;base64,/9j/4AAQ...');
    expect(result.previews.unet_overlay).toBe('data:image/jpeg;base64,/9j/4AAQ...');
  });

  it('should preserve overlay segmentation objects', () => {
    const scan = { id: '123' };
    const segmentation = { overlay: 'https://hf.space/file=overlay.png' };
    const result = normalizeScanResult(scan, segmentation);
    expect(result.segmentation).toEqual({ overlay: 'https://hf.space/file=overlay.png' });
  });
});

