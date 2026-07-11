"use client";
import React, { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [scan, setScan] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [uploadState, setUploadState] = useState({ status: "Waiting", progress: 0, fileName: "", error: "" });
  const [decision, setDecision] = useState({ choice: "", rationale: "", submittedAt: "" });
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const savedScan = localStorage.getItem("oct_scan");
    const savedHistory = localStorage.getItem("oct_scan_history");
    const savedUploadState = localStorage.getItem("oct_uploadState");
    const savedDecision = localStorage.getItem("oct_decision");

    if (savedScan) {
        try { setScan(JSON.parse(savedScan)); } catch (e) {}
    }
    if (savedHistory) {
        try { setScanHistory(JSON.parse(savedHistory)); } catch (e) {}
    }
    if (savedUploadState) {
        try { setUploadState(JSON.parse(savedUploadState)); } catch (e) {}
    }
    if (savedDecision) {
        try { setDecision(JSON.parse(savedDecision)); } catch (e) {}
    }
    setIsLoaded(true);
  }, []);

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

  const addScanToHistory = (newScan) => {
    setScanHistory(prev => {
      // Check if it already exists by some ID, or just append. 
      // If patient_id is not unique (can have multiple scans), we should ideally assign a unique ID.
      // We can use scan.id or generate one. The backend returns an 'id' in create_scan (like "b9d01cf0f18d9ce1").
      const exists = prev.find(s => s.id === newScan.id);
      if (exists) {
        return prev.map(s => s.id === newScan.id ? newScan : s);
      }
      return [newScan, ...prev];
    });
  };

  const deleteScan = (id) => {
    setScanHistory(prev => prev.filter(s => s.id !== id));
    if (scan && scan.id === id) {
      setScan(null);
      setUploadState({ status: "Waiting", progress: 0, fileName: "", error: "" });
      setDecision({ choice: "", rationale: "", submittedAt: "" });
    }
  };

  return (
    <AppContext.Provider value={{
      scan, setScan,
      scanHistory, setScanHistory,
      addScanToHistory, deleteScan,
      uploadState, setUploadState,
      decision, setDecision
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  return useContext(AppContext);
}
