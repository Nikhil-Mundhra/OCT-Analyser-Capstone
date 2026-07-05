"use client";
import React, { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [scan, setScan] = useState(null);
  const [uploadState, setUploadState] = useState({ status: "Waiting", progress: 0, fileName: "", error: "" });
  const [decision, setDecision] = useState({ choice: "", rationale: "", submittedAt: "" });
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const savedScan = localStorage.getItem("oct_scan");
    const savedUploadState = localStorage.getItem("oct_uploadState");
    const savedDecision = localStorage.getItem("oct_decision");

    if (savedScan) {
        try { setScan(JSON.parse(savedScan)); } catch (e) {}
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
    localStorage.setItem("oct_uploadState", JSON.stringify(uploadState));
  }, [uploadState, isLoaded]);

  useEffect(() => {
    if (!isLoaded) return;
    localStorage.setItem("oct_decision", JSON.stringify(decision));
  }, [decision, isLoaded]);

  return (
    <AppContext.Provider value={{
      scan, setScan,
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
