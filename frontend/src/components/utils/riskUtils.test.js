import { describe, it, expect } from 'vitest';
import { riskFromScan } from './riskUtils';

describe('riskFromScan', () => {
  it('should return default values when scan is null or not completed', () => {
    const resultNull = riskFromScan(null);
    expect(resultNull.label).toBe('No active scan');
    expect(resultNull.tone).toBe('neutral');

    const resultIncomplete = riskFromScan({ status: 'processing' });
    expect(resultIncomplete.label).toBe('No active scan');
  });

  it('should identify high risk diagnoses', () => {
    const scan = { status: 'completed', diagnosis: 'AMD_WET', confidence: 0.95 };
    const result = riskFromScan(scan);
    expect(result.label).toBe('High risk');
    expect(result.tone).toBe('danger');
    expect(result.confidence).toBe('95%');
  });

  it('should identify moderate risk diagnoses', () => {
    const scan = { status: 'completed', diagnosis: 'MH', confidence: 0.8 };
    const result = riskFromScan(scan);
    expect(result.label).toBe('Ambiguous');
    expect(result.tone).toBe('warning');
  });

  it('should treat low confidence as ambiguous even for safe diagnoses', () => {
    const scan = { status: 'completed', diagnosis: 'NORMAL', confidence: 0.6 };
    const result = riskFromScan(scan);
    expect(result.label).toBe('Ambiguous');
    expect(result.tone).toBe('warning');
  });

  it('should identify low risk for safe diagnoses with high confidence', () => {
    const scan = { status: 'completed', diagnosis: 'NORMAL', confidence: 0.9 };
    const result = riskFromScan(scan);
    expect(result.label).toBe('Low risk');
    expect(result.tone).toBe('safe');
  });
});
