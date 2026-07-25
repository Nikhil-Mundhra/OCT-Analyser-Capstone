"use client";
import React, { useCallback, useState } from "react";
import { usePathname } from "next/navigation";
import { ChevronDown, ChevronUp, UserSquare2 } from "lucide-react";
import { ROUTES } from "../constants";
import { useAppContext } from "../../AppContext";

const VALID_PATHS = [ROUTES.QC, ROUTES.REVIEW, ROUTES.DECISION];

export function CaseSwitcher() {
  const { scan, scanHistory, setScan, setUploadState } = useAppContext();
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  // All hooks must be called before any early return (Rules of Hooks)
  const handleSelect = useCallback((selectedScan) => {
    setScan(selectedScan);
    setUploadState({ status: "Completed", progress: 100, fileName: selectedScan.filename || "", error: "" });
    setIsOpen(false);
  }, [setScan, setUploadState]);

  // Only render on clinical tabs where case-switching is meaningful
  if (!VALID_PATHS.includes(pathname)) return null;

  return (
    <div className="relative z-20">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
      >
        <UserSquare2 className="h-4 w-4 text-sky-500" />
        {scan?.patient_id ? `Active: ${scan.patient_id}` : "Select Patient"}
        {isOpen ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-64 rounded-2xl border border-slate-200 bg-white p-2 shadow-lg">
          {scanHistory.length === 0 ? (
            <div className="p-3 text-sm text-slate-500 text-center">No previous scans found.</div>
          ) : (
            <div className="flex flex-col gap-1 max-h-60 overflow-y-auto">
              {scanHistory.map((s) => (
                <button
                  key={s.scan_id}
                  onClick={() => handleSelect(s)}
                  className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors text-left ${
                    scan?.scan_id === s.scan_id ? "bg-sky-50 text-sky-700 font-bold" : "text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <div>
                    <div className="font-bold">{s.patient_id || "Unknown"}</div>
                    <div className="text-xs opacity-70">{s.scan_type || s.source_format}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
