
export function CaseSwitcher() {
  const { scan, scanHistory, setScan, setUploadState, setDecision } = useAppContext();
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  // Only show on these specific clinical tabs
  const validPaths = ["/QC", "/review", "/human-check"];
  if (!validPaths.includes(pathname)) return null;

  const handleSelect = (selectedScan) => {
    setScan(selectedScan);
    setUploadState({ status: "Completed", progress: 100, fileName: selectedScan.filename || "", error: "" });
    setDecision({ choice: "", rationale: "", submittedAt: "" });
    setIsOpen(false);
  };

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
                  key={s.id}
                  onClick={() => handleSelect(s)}
                  className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors text-left ${
                    scan?.id === s.id ? "bg-sky-50 text-sky-700 font-bold" : "text-slate-700 hover:bg-slate-50"
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
