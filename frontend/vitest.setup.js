import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Mock localStorage
const localStorageMock = (function () {
  let store = {};
  return {
    getItem(key) {
      return store[key] || null;
    },
    setItem(key, value) {
      store[key] = value.toString();
    },
    removeItem(key) {
      delete store[key];
    },
    clear() {
      store = {};
    }
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true
});

// Cleanup after each test case
afterEach(() => {
  cleanup();
  window.localStorage.clear();
});
