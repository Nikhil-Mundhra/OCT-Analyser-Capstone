import React, { useState } from 'react';
import { runModelSuite } from '../../api/octAnalyzerClient';
import ExportPreferencesModal from '../modals/ExportPreferencesModal';

const MODEL_CONFIGS = [
  { id: 'all', name: 'Master Suite (All 5)', badge: 'Complete Suite', metric: 'Full Biomarker Pipeline' },
  { id: 'model1', name: 'Retinal Layers U-Net', badge: 'mDice: 0.9452', metric: '6-Class Layer Boundaries' },
  { id: 'model2', name: 'Choroidalyzer U-Net', badge: 'Dice: 0.9610', metric: 'Choroid Region & Thickness' },
  { id: 'model3', name: 'HRF Attention U-Net', badge: 'Fluid Dice: 0.9380', metric: 'Fluid Accumulation & DME' },
  { id: 'model4', name: 'OIMHS Hole & Cyst U-Net', badge: 'Dice: 0.9701', metric: 'Macular Hole & Cysts (IRC)' },
  { id: 'model5', name: 'OCT Pathology Detector', badge: 'mAP@0.5: 0.8650', metric: '9-Class Object Detector' },
];

export default function SegmentationSuitePanel({ file, classification }) {
  const [selectedModel, setSelectedModel] = useState('all');
  const [activeTab, setActiveTab] = useState('model1');
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [suiteResults, setSuiteResults] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.85);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  const handleRunInference = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await runModelSuite(file, selectedModel, threshold);
      if (res && res.results) {
        setSuiteResults(res.results);
        const keys = Object.keys(res.results);
        if (keys.length > 0) setActiveTab(keys[0]);
      }
    } catch (err) {
      console.error('Error running segmentation suite:', err);
      alert('Failed to run segmentation suite: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const currentResult = suiteResults ? suiteResults[activeTab] : null;

  return (
    <div style={{ background: '#0F172A', color: '#F8FAFC', padding: '24px', borderRadius: '16px', border: '1px solid #1E293B', marginBottom: '24px' }}>
      {/* Header */}
      <header style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '1.35rem', margin: 0, color: '#38BDF8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            Segmentation 5-Model Suite &amp; Diagnostic Engine
          </h2>
          <p style={{ fontSize: '0.85rem', color: '#94A3B8', margin: '4px 0 0' }}>
            Selective or full-suite deep learning inference for layer boundaries, choroidal thickness, fluid attention, macular cysts, &amp; pathology bounding boxes.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            onClick={() => setIsExportModalOpen(true)}
            disabled={!file && !classification && !suiteResults}
            style={{
              background: '#1E293B',
              color: '#38BDF8',
              padding: '10px 18px',
              borderRadius: '8px',
              border: '1px solid #0284C7',
              fontWeight: '600',
              fontSize: '0.85rem',
              cursor: !file && !classification && !suiteResults ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            Export Diagnostic Report (ZIP)
          </button>
          <button
            onClick={handleRunInference}
            disabled={loading || !file}
            style={{
              background: loading ? '#475569' : 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)',
              color: '#FFF',
              padding: '10px 22px',
              borderRadius: '8px',
              border: 'none',
              fontWeight: '600',
              fontSize: '0.9rem',
              cursor: loading || !file ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 12px rgba(2, 132, 199, 0.3)',
              transition: 'all 0.2s ease',
            }}
          >
            {loading ? 'Processing Suite...' : 'Run Segmentation Suite'}
          </button>
        </div>
      </header>

      {/* Dual-Head Classification Summary Card (Level 1 Screening + Level 2 Router) */}
      {classification && (() => {
        const l1 = classification.Level1 || classification.level1 || classification.pipeline_results?.Level1;
        const l2 = classification.Level2 || classification.level2 || classification.pipeline_results?.Level2;
        
        const l1Label = l1?.prediction || l1?.label || (classification.diagnosis === 'NORMAL' ? 'NORMAL' : 'ABNORMAL');
        const l1Conf = l1?.confidence != null ? `${(l1.confidence * 100).toFixed(1)}%` : (classification.confidence ? `${(classification.confidence * 100).toFixed(1)}%` : 'N/A');
        
        const l2Label = l2?.prediction || l2?.label || classification.diagnosis || 'Pathology Detected';
        const l2Conf = l2?.confidence != null ? `${(l2.confidence * 100).toFixed(1)}%` : (classification.confidence ? `${(classification.confidence * 100).toFixed(1)}%` : 'N/A');

        return (
          <div style={{ background: '#1E293B', borderRadius: '12px', padding: '16px', marginBottom: '20px', border: '1px solid #334155' }}>
            <h3 style={{ fontSize: '1rem', color: '#38BDF8', margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
              Dual-Head Diagnostic Impression (ConvNeXt V2)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
              {/* Level 1 Binary Health Screening */}
              <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px', borderLeft: `4px solid ${l1Label === 'NORMAL' ? '#4ADE80' : '#F87171'}` }}>
                <div style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', fontWeight: 'bold' }}>Level 1: Health Screening</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                  <span style={{ fontSize: '1rem', fontWeight: 'bold', color: l1Label === 'NORMAL' ? '#4ADE80' : '#F87171' }}>
                    {l1Label}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
                    {l1Conf}
                  </span>
                </div>
              </div>

              {/* Level 2 15-Class Granular Disease Router */}
              <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #818CF8' }}>
                <div style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', fontWeight: 'bold' }}>Level 2: Granular Disease Router</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                  <span style={{ fontSize: '0.95rem', fontWeight: 'bold', color: '#E2E8F0' }}>
                    {l2Label}
                  </span>
                  <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
                    {l2Conf}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Model Selection Selector */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px', marginBottom: '20px' }}>
        {MODEL_CONFIGS.map((m) => (
          <div key={m.id} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button
              onClick={() => setSelectedModel(m.id)}
              style={{
                padding: '12px',
                borderRadius: '10px',
                border: selectedModel === m.id ? '2px solid #38BDF8' : '1px solid #334155',
                background: selectedModel === m.id ? '#1E293B' : '#0F172A',
                color: '#F8FAFC',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                height: '100%',
              }}
            >
              <div style={{ fontSize: '0.85rem', fontWeight: 'bold', color: selectedModel === m.id ? '#38BDF8' : '#F8FAFC' }}>
                {m.name}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', marginTop: '4px' }}>{m.badge}</div>
              <div style={{ fontSize: '0.7rem', color: '#64748B', marginTop: '2px' }}>{m.metric}</div>
            </button>

            {/* Baked Threshold Slider inside Model 5 Card */}
            {m.id === 'model5' && (selectedModel === 'model5' || selectedModel === 'all') && (
              <div style={{ background: '#020617', padding: '8px 10px', borderRadius: '8px', border: '1px solid #1E293B', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#94A3B8', fontWeight: 'bold' }}>
                  <span>Score Threshold:</span>
                  <strong style={{ color: '#38BDF8', fontFamily: 'monospace' }}>{threshold}</strong>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: '#0284C7', cursor: 'pointer', height: '4px' }}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Output Viewer & Metrics */}
      {suiteResults && (
        <div>
          {/* Sub-tabs for executed models */}
          <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid #334155', paddingBottom: '8px', marginBottom: '16px', overflowX: 'auto' }}>
            {Object.keys(suiteResults).map((key) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  background: activeTab === key ? '#0284C7' : '#1E293B',
                  color: '#FFF',
                  fontWeight: '600',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {suiteResults[key].name}
              </button>
            ))}
          </div>

          {currentResult && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
              {/* Overlay Image Canvas */}
              <div style={{ background: '#020617', padding: '16px', borderRadius: '12px', border: '1px solid #1E293B', textAlign: 'center' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontSize: '0.8rem', color: '#94A3B8', fontWeight: 'bold' }}>Segmentation Overlay</span>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center', fontSize: '0.8rem', color: '#94A3B8' }}>
                    <span>Opacity:</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                      style={{ width: '90px' }}
                    />
                  </div>
                </div>
                {currentResult.overlay ? (
                  <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', borderRadius: '8px', overflow: 'hidden' }}>
                    {/* Layer 1: Untouched Background OCT Scan at 100% Opacity */}
                    {file ? (
                      <img
                        src={URL.createObjectURL(file)}
                        alt="Raw OCT Scan Background"
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '420px', borderRadius: '8px' }}
                      />
                    ) : (
                      <img
                        src={currentResult.overlay}
                        alt="Background OCT Scan"
                        style={{ display: 'block', maxWidth: '100%', maxHeight: '420px', borderRadius: '8px' }}
                      />
                    )}
                    
                    {/* Layer 2: Mask Overlay with Dynamic Opacity Slider Control */}
                    <img
                      src={currentResult.mask || currentResult.overlay}
                      alt="Segmentation Mask Overlay"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        borderRadius: '8px',
                        opacity: overlayOpacity,
                        mixBlendMode: currentResult.mask ? 'normal' : 'screen',
                        pointerEvents: 'none',
                      }}
                    />
                  </div>
                ) : (
                  <div style={{ padding: '40px', color: '#64748B', fontSize: '0.9rem' }}>No overlay generated.</div>
                )}
              </div>

              {/* Biomarker Metrics Sidebar */}
              <div style={{ background: '#1E293B', padding: '16px', borderRadius: '12px', border: '1px solid #334155' }}>
                <h3 style={{ fontSize: '1rem', color: '#38BDF8', margin: '0 0 10px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  Quantitative Biomarker Metrics
                </h3>
                <pre style={{
                  background: '#0F172A',
                  padding: '14px',
                  borderRadius: '8px',
                  fontSize: '0.85rem',
                  color: '#E2E8F0',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: '340px',
                  overflowY: 'auto',
                  fontFamily: 'monospace',
                  lineHeight: '1.5',
                  border: '1px solid #1E293B',
                }}>
                  {currentResult.details || 'No metrics returned.'}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Export Customization Preferences Modal */}
      <ExportPreferencesModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        file={file}
        scan={classification}
        suiteResults={suiteResults}
      />
    </div>
  );
}
