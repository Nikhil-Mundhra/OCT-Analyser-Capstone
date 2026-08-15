import React, { useState, useEffect } from 'react';
import { exportDiagnosticPackage } from '../../utils/reportExporter';

const PREF_STORAGE_KEY = 'oct_export_preferences';

export default function ExportPreferencesModal({ isOpen, onClose, file, scan, suiteResults }) {
  const [options, setOptions] = useState({
    includePdf: true,
    includeRawScan: true,
    includeModel1: true,
    includeModel2: true,
    includeModel3: true,
    includeModel4: true,
    includeModel5: true,
  });

  const [exporting, setExporting] = useState(false);
  const [smartNotice, setSmartNotice] = useState('');

  useEffect(() => {
    if (!isOpen) return;

    // Determine smart defaults based on classification
    const diagnosis = (
      scan?.Level2?.prediction ||
      scan?.level2?.label ||
      scan?.diagnosis ||
      ''
    ).toUpperCase();

    let smartDefaults = {
      includePdf: true,
      includeRawScan: true,
      includeModel1: true,
      includeModel2: true,
      includeModel3: true,
      includeModel4: true,
      includeModel5: true,
    };

    let notice = '';

    if (diagnosis.includes('MH') || diagnosis.includes('HOLE') || diagnosis.includes('VMT')) {
      smartDefaults.includeModel2 = false; // Omit Choroidalyzer for Macular Holes due to dip breakdown
      smartDefaults.includeModel4 = true;  // Highlight OIMHS Hole/Cyst model
      notice = 'Smart Recommendation: Model 2 (Choroidalyzer) has been pre-deselected because Macular Hole pathology disrupts the foveal contour.';
    } else if (diagnosis.includes('DME') || diagnosis.includes('DR') || diagnosis.includes('RVO')) {
      smartDefaults.includeModel4 = false;
      notice = 'Smart Recommendation: Model 3 (HRF DME Fluid Attention) is highlighted for diabetic & vascular fluid accumulation.';
    } else if (diagnosis === 'NORMAL') {
      smartDefaults.includeModel2 = false;
      smartDefaults.includeModel3 = false;
      smartDefaults.includeModel4 = false;
      smartDefaults.includeModel5 = false;
      notice = 'Smart Recommendation: Normal scan detected. Retinal Layer UNet (Model 1) is selected.';
    }

    setSmartNotice(notice);

    // Try restoring saved preferences from localStorage ("cookie preferences")
    try {
      const saved = localStorage.getItem(PREF_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        setOptions((prev) => ({ ...prev, ...smartDefaults, ...parsed }));
      } else {
        setOptions(smartDefaults);
      }
    } catch (e) {
      setOptions(smartDefaults);
    }
  }, [isOpen, scan]);

  if (!isOpen) return null;

  const handleToggle = (key) => {
    const next = { ...options, [key]: !options[key] };
    setOptions(next);
    try {
      localStorage.setItem(PREF_STORAGE_KEY, JSON.stringify(next));
    } catch (e) {
      console.warn('Failed to save export preferences:', e);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      await exportDiagnosticPackage({ file, scan, suiteResults, options });
      onClose();
    } catch (err) {
      console.error('Export failed:', err);
      alert('Export failed: ' + err.message);
    } finally {
      setExporting(false);
    }
  };

  const items = [
    { key: 'includePdf', label: 'Diagnostic Evaluation Report (PDF)', desc: 'Formatted multi-page medical report with ConvNeXt V2 impressions and tables.' },
    { key: 'includeRawScan', label: 'Raw OCT Input Image (PNG)', desc: 'Untouched high-resolution original OCT scan.' },
    { key: 'includeModel1', label: 'Model 1: Retinal Layers Mask', desc: '6-class retinal layer boundary overlay.' },
    { key: 'includeModel2', label: 'Model 2: Choroidalyzer Mask', desc: 'Choroidal region & thickness mask (Omitted for Macular Hole scans).' },
    { key: 'includeModel3', label: 'Model 3: HRF DME Fluid Mask', desc: 'High-resolution fluid accumulation & DME attention mask.' },
    { key: 'includeModel4', label: 'Model 4: OIMHS Hole & Cyst Mask', desc: 'Macular hole & intraretinal cyst (IRC) mask.' },
    { key: 'includeModel5', label: 'Model 5: Pathology Detector Overlay', desc: '9-class biomarker object detector bounding boxes.' },
  ];

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(2, 6, 23, 0.85)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      padding: '20px'
    }}>
      <div style={{
        background: '#0F172A',
        border: '1px solid #1E293B',
        borderRadius: '16px',
        maxWidth: '560px',
        width: '100%',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
        padding: '24px',
        color: '#F8FAFC'
      }}>
        {/* Modal Header */}
        <header style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', margin: 0, color: '#38BDF8', fontWeight: 'bold' }}>
              Export Diagnostic Package
            </h2>
            <p style={{ fontSize: '0.8rem', color: '#94A3B8', margin: '4px 0 0' }}>
              Customize the files to include in your downloadable ZIP package.
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: '#64748B', fontSize: '1.5rem', cursor: 'pointer' }}
          >
            &times;
          </button>
        </header>

        {/* Smart Classification Recommendation Banner */}
        {smartNotice && (
          <div style={{
            background: 'rgba(2, 132, 199, 0.1)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '8px',
            padding: '10px 14px',
            marginBottom: '16px',
            fontSize: '0.8rem',
            color: '#38BDF8',
            lineHeight: '1.4'
          }}>
            {smartNotice}
          </div>
        )}

        {/* Form Preferences List */}
        <div style={{ maxHeight: '320px', overflowY: 'auto', marginBottom: '20px', paddingRight: '4px' }}>
          {items.map((item) => (
            <label
              key={item.key}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '12px',
                padding: '10px 12px',
                borderRadius: '8px',
                background: options[item.key] ? '#1E293B' : 'transparent',
                border: options[item.key] ? '1px solid #334155' : '1px solid transparent',
                cursor: 'pointer',
                marginBottom: '6px',
                transition: 'all 0.15s ease'
              }}
            >
              <input
                type="checkbox"
                checked={!!options[item.key]}
                onChange={() => handleToggle(item.key)}
                style={{ marginTop: '3px', accentColor: '#0284C7', width: '16px', height: '16px', cursor: 'pointer' }}
              />
              <div>
                <div style={{ fontSize: '0.85rem', fontWeight: 'bold', color: options[item.key] ? '#F8FAFC' : '#94A3B8' }}>
                  {item.label}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748B', marginTop: '2px' }}>
                  {item.desc}
                </div>
              </div>
            </label>
          ))}
        </div>

        {/* Actions Footer */}
        <footer style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button
            onClick={onClose}
            disabled={exporting}
            style={{
              background: '#1E293B',
              color: '#94A3B8',
              padding: '10px 18px',
              borderRadius: '8px',
              border: '1px solid #334155',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={exporting}
            style={{
              background: exporting ? '#475569' : 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)',
              color: '#FFF',
              padding: '10px 22px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: exporting ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 12px rgba(2, 132, 199, 0.4)'
            }}
          >
            {exporting ? 'Compiling ZIP...' : 'Download ZIP Package'}
          </button>
        </footer>
      </div>
    </div>
  );
}
