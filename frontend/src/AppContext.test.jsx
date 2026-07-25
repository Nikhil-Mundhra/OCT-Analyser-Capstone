import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { AppProvider, useAppContext } from './AppContext';

const TestComponent = () => {
  const {
    isLoaded,
    scanHistory,
    uploadState,
    decision,
    addScanToHistory,
    resetUpload,
    completeDecision
  } = useAppContext();

  return (
    <div>
      <div data-testid="isLoaded">{isLoaded ? 'true' : 'false'}</div>
      <div data-testid="history-length">{scanHistory.length}</div>
      <div data-testid="upload-status">{uploadState.status}</div>
      <div data-testid="decision-choice">{decision.choice || 'none'}</div>

      <button onClick={() => addScanToHistory({ scan_id: '1', name: 'Scan 1' })}>
        Add Scan
      </button>
      <button onClick={() => resetUpload()}>
        Reset Upload
      </button>
      <button onClick={() => completeDecision('Agree', 'Looks good')}>
        Complete Decision
      </button>
    </div>
  );
};

describe('AppContext', () => {
  it('should provide default values initially', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    expect(screen.getByTestId('isLoaded').textContent).toBe('true');
    expect(screen.getByTestId('history-length').textContent).toBe('0');
    expect(screen.getByTestId('upload-status').textContent).toBe('Waiting');
    expect(screen.getByTestId('decision-choice').textContent).toBe('none');
  });

  it('should handle addScanToHistory action', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    act(() => {
      screen.getByText('Add Scan').click();
    });

    expect(screen.getByTestId('history-length').textContent).toBe('1');
  });

  it('should handle resetUpload action', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    act(() => {
      screen.getByText('Reset Upload').click();
    });

    expect(screen.getByTestId('upload-status').textContent).toBe('Waiting');
    expect(screen.getByTestId('decision-choice').textContent).toBe('none');
  });

  it('should handle completeDecision action', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    act(() => {
      screen.getByText('Complete Decision').click();
    });

    expect(screen.getByTestId('decision-choice').textContent).toBe('Agree');
  });
});
