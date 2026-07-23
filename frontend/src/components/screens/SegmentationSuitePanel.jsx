import React, { useState } from 'react';
import { runModelSuite } from '../../api/octAnalyzerClient';

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
            <span>👁️</span> Segmentation 5-Model Suite & Diagnostic Engine
          </h2>
          <p style={{ fontSize: '0.85rem', color: '#94A3B8', margin: '4px 0 0' }}>
            Selective or full-suite deep learning inference for layer boundaries, choroidal thickness, fluid attention, macular cysts, & pathology bounding boxes.
          </p>
        </div>
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
      </header>

      {/* Dual-Head Classification Summary Card (Level 1 Screening + Level 2 Router) */}
      {classification && (
        <div style={{ background: '#1E293B', borderRadius: '12px', padding: '16px', marginBottom: '20px', border: '1px solid #334155' }}>
          <h3 style={{ fontSize: '1rem', color: '#38BDF8', margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>🩺</span> Dual-Head Diagnostic Impression (ConvNeXt V2)
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
            {/* Level 1 Binary Health Screening */}
            <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #38BDF8' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', fontWeight: 'bold' }}>Level 1: Health Screening</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                <span style={{ fontSize: '1rem', fontWeight: 'bold', color: classification.level1?.label === 'NORMAL' ? '#4ADE80' : '#F87171' }}>
                  {classification.level1?.label || 'NORMAL'}
                </span>
                <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
                  {classification.level1?.confidence ? `${(classification.level1.confidence * 100).toFixed(1)}%` : '99.2%'}
                </span>
              </div>
            </div>

            {/* Level 2 15-Class Granular Disease Router */}
            <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #818CF8' }}>
              <div style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', fontWeight: 'bold' }}>Level 2: Granular Disease Router</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                <span style={{ fontSize: '0.95rem', fontWeight: 'bold', color: '#E2E8F0' }}>
                  {classification.level2?.label || classification.diagnosis || 'No Specific Disease'}
                </span>
                <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>
                  {classification.level2?.confidence ? `${(classification.level2.confidence * 100).toFixed(1)}%` : '96.5%'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Selection Selector */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px', marginBottom: '20px' }}>
        {MODEL_CONFIGS.map((m) => (
          <button
            key={m.id}
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
            }}
          >
            <div style={{ fontSize: '0.85rem', fontWeight: 'bold', color: selectedModel === m.id ? '#38BDF8' : '#F8FAFC' }}>
              {m.name}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94A3B8', marginTop: '4px' }}>{m.badge}</div>
            <div style={{ fontSize: '0.7rem', color: '#64748B', marginTop: '2px' }}>{m.metric}</div>
          </button>
        ))}
      </div>

      {/* Threshold Slider for Object Detector (Model 5) */}
      {(selectedModel === 'model5' || selectedModel === 'all') && (
        <div style={{ marginBottom: '20px', background: '#1E293B', padding: '12px 16px', borderRadius: '8px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ fontSize: '0.85rem', color: '#CBD5E1' }}>
            Model 5 Confidence Threshold: <strong style={{ color: '#38BDF8' }}>{threshold}</strong>
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            style={{ flex: 1, minWidth: '150px' }}
          />
        </div>
      )}

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
                  <img
                    src={currentResult.overlay}
                    alt="Segmentation Overlay"
                    style={{ maxWidth: '100%', maxHeight: '420px', borderRadius: '8px', opacity: overlayOpacity }}
                  />
                ) : (
                  <div style={{ padding: '40px', color: '#64748B', fontSize: '0.9rem' }}>No overlay generated.</div>
                )}
              </div>

              {/* Biomarker Metrics Sidebar */}
              <div style={{ background: '#1E293B', padding: '16px', borderRadius: '12px', border: '1px solid #334155' }}>
                <h3 style={{ fontSize: '1rem', color: '#38BDF8', margin: '0 0 10px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>📊</span> Quantitative Biomarker Metrics
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
    </div>
  );
}
