import React, { createContext, useContext, useState } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [scan, setScan] = useState(null);
  const [uploadState, setUploadState] = useState({ status: "Waiting", progress: 0, fileName: "", error: "" });
  const [decision, setDecision] = useState({ choice: "", rationale: "", submittedAt: "" });

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
