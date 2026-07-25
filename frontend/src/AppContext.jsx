"use client";
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AppContext = createContext(null);

/** Shape of the uploadState reset value — single source of truth. */
export const INITIAL_UPLOAD_STATE = { status: "Waiting", progress: 0, fileName: "", error: "" };

/** Shape of the decision reset value — single source of truth. */
export const INITIAL_DECISION = { choice: "", rationale: "", submittedAt: "" };

export function AppProvider({ children }) {
  const [scan, setScan] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [uploadState, setUploadState] = useState(INITIAL_UPLOAD_STATE);
  const [decision, setDecision] = useState(INITIAL_DECISION);
  const [isLoaded, setIsLoaded] = useState(false);

  // Rehydrate from localStorage once on mount
  useEffect(() => {
    const savedScan = localStorage.getItem("oct_scan");
    const savedHistory = localStorage.getItem("oct_scan_history");
    const savedUploadState = localStorage.getItem("oct_uploadState");
    const savedDecision = localStorage.getItem("oct_decision");

    if (savedScan) {
      try { setScan(JSON.parse(savedScan)); } catch (e) { console.error('[AppContext] Failed to parse saved scan:', e); }
    }
    if (savedHistory) {
      try { setScanHistory(JSON.parse(savedHistory)); } catch (e) { console.error('[AppContext] Failed to parse scan history:', e); }
    }
    if (savedUploadState) {
      try { setUploadState(JSON.parse(savedUploadState)); } catch (e) { console.error('[AppContext] Failed to parse upload state:', e); }
    }
    if (savedDecision) {
      try { setDecision(JSON.parse(savedDecision)); } catch (e) { console.error('[AppContext] Failed to parse decision:', e); }
    }
    setIsLoaded(true);
  }, []);

  // Persist to localStorage whenever state changes (after hydration)
  useEffect(() => {
    if (!isLoaded) return;
    if (scan) localStorage.setItem("oct_scan", JSON.stringify(scan));
    else localStorage.removeItem("oct_scan");
  }, [scan, isLoaded]);

  useEffect(() => {
    if (!isLoaded) return;
    localStorage.setItem("oct_scan_history", JSON.stringify(scanHistory));
  }, [scanHistory, isLoaded]);

  useEffect(() => {
    if (!isLoaded) return;
    localStorage.setItem("oct_uploadState", JSON.stringify(uploadState));
  }, [uploadState, isLoaded]);

  useEffect(() => {
    if (!isLoaded) return;
    localStorage.setItem("oct_decision", JSON.stringify(decision));
  }, [decision, isLoaded]);

  // --- Named actions (consumers use these; raw setters for uploadState/decision are not exposed) ---

  const addScanToHistory = useCallback((newScan) => {
    setScanHistory(prev => {
      // Use scan_id (set by normalizeScanResult) as the canonical dedup key
      const canonicalId = newScan.scan_id || newScan.id;
      const exists = prev.find(s => (s.scan_id || s.id) === canonicalId);
      if (exists) {
        return prev.map(s => (s.scan_id || s.id) === canonicalId ? newScan : s);
      }
      return [newScan, ...prev];
    });
  }, []);

  const deleteScan = useCallback((id) => {
    setScanHistory(prev => prev.filter(s => (s.scan_id || s.id) !== id));
    if (scan && (scan.scan_id || scan.id) === id) {
      setScan(null);
      setUploadState(INITIAL_UPLOAD_STATE);
      setDecision(INITIAL_DECISION);
    }
  }, [scan]);

  /** Reset upload + decision state to defaults. Call when starting a new upload. */
  const resetUpload = useCallback(() => {
    setScan(null);
    setUploadState(INITIAL_UPLOAD_STATE);
    setDecision(INITIAL_DECISION);
  }, []);

  /** Persist a finalised clinician decision. */
  const completeDecision = useCallback((choice, rationale) => {
    setDecision({ choice, rationale, submittedAt: new Date().toLocaleString() });
  }, []);

  return (
    <AppContext.Provider value={{
      // State (read-only consumers should only use these)
      scan,
      scanHistory,
      uploadState,
      decision,
      isLoaded,
      // Escape hatches — raw setters intentionally kept for direct scan assignment after API
      setScan,
      setUploadState,
      // Named actions — prefer these over raw setters
      addScanToHistory,
      deleteScan,
      resetUpload,
      completeDecision,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('[useAppContext] Must be called within an <AppProvider>. Did you forget to wrap your component tree?');
  }
  return ctx;
}
