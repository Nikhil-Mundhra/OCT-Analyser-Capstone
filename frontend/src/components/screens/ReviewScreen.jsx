"use client";
import React, { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Brain,
  ScanLine,
} from "lucide-react";
import { useAppContext } from "../../AppContext";
import { Card } from "../ui/Card";
import { Metric } from "../ui/Metric";
import { WireBox } from "../ui/WireBox";
import { riskFromScan } from "../utils/riskUtils";
import { getClassColor } from "../utils/colorUtils";

export function ReviewScreen() {
  const { scan, resetUpload } = useAppContext();
  const router = useRouter();
  const [viewMode, setViewMode] = useState("segmented");

  const handleReset = useCallback(() => {
    resetUpload();
    router.push("/QC");
  }, [resetUpload, router]);

  const completed = scan?.status === "completed";
  const risk = riskFromScan(scan);

  const l1 = scan?.level1;
  const l2 = scan?.level2;
  const l3 = scan?.level3;
  const l1Abnormal = l1?.prediction?.toUpperCase() === "ABNORMAL";
  const gradcams = scan?.gradcams;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex justify-end">
        <button onClick={handleReset} className="rounded-2xl bg-rose-100 px-4 py-2 text-sm font-bold text-rose-700 hover:bg-rose-200 transition-colors">
          Delete Scan &amp; Start New
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-12">
        <Card title="OCT Image Classification" subtitle="Hugging Face &amp; Local Segmentation" icon={ScanLine} className="lg:col-span-7">
          <div className="grid gap-4">
            <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 flex flex-col justify-between" style={{ minHeight: "24rem" }}>
              {completed && scan.localImageUrl ? (
                <>
                  <div className="absolute inset-0 flex items-center justify-center p-4">
                    <div className="relative max-h-full max-w-full">
                      <img src={scan.localImageUrl} alt="Uploaded OCT" className="max-h-full max-w-full block rounded" />
                      {viewMode === "segmented" && scan.segmentation && (
                        <svg viewBox="0 0 512 512" className="absolute top-0 left-0 h-full w-full pointer-events-none" preserveAspectRatio="none">
                          {(scan.segmentation.layers || []).map((layer, idx) => (
                            <polyline
                              key={`layer-${idx}`}
                              points={layer.boundary_points.map(p => `${p.x},${p.y}`).join(" ")}
                              fill="none"
                              stroke={getClassColor(layer.class_name).stroke}
                              strokeWidth="2"
                            />
                          ))}
                          {(scan.segmentation.lesions || []).map((lesion, idx) => {
                            const color = getClassColor(lesion.class_name);
                            return (
                              <polygon
                                key={`lesion-${idx}`}
                                points={lesion.polygon.map(p => `${p.x},${p.y}`).join(" ")}
                                fill={color.fill}
                                stroke={color.stroke}
                                strokeWidth="2"
                              />
                            );
                          })}
                        </svg>
                      )}
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-0 right-0 flex justify-center z-10">
                    <div className="flex bg-slate-800/80 backdrop-blur-md rounded-xl p-1 gap-1 border border-slate-700">
                      <button
                        onClick={() => setViewMode("raw")}
                        className={`px-4 py-1.5 rounded-lg text-xs font-bold transition ${viewMode === "raw" ? "bg-white text-slate-900" : "text-slate-300 hover:text-white"}`}
                      >
                        Raw
                      </button>
                      <button
                        onClick={() => setViewMode("segmented")}
                        className={`px-4 py-1.5 rounded-lg text-xs font-bold transition ${viewMode === "segmented" ? "bg-white text-slate-900" : "text-slate-300 hover:text-white"}`}
                      >
                        Segmented
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <WireBox height="h-72">Upload a scan to view predictions</WireBox>
              )}
            </div>
          </div>
        </Card>

        <div className="lg:col-span-5 space-y-5">
          {scan?.ai_supported === false ? (
            <Card title="AI Inference Disabled" subtitle="Incompatible scan parameters" icon={AlertTriangle}>
              <div className="rounded-2xl bg-amber-50 p-5 text-sm text-amber-900 border border-amber-200 shadow-sm">
                <div className="font-bold flex items-center gap-2 mb-2 text-amber-950 text-base">
                  <AlertTriangle className="h-5 w-5" /> Unsupported Modality
                </div>
                The current AI model suite is only validated for <b>Structural OCT</b> scans of the <b>Macula</b>.
                <br /><br />
                The scan you uploaded ({scan?.modality} - {scan?.target}) cannot be processed because the anatomical structures or imaging physics do not match the training distribution.
                <br /><br />
                You can still use this interface to view the scan, store it in the patient&apos;s timeline, or write manual clinical notes.
              </div>
            </Card>
          ) : (
            <>
              <Card title="Level 1: Triage (Hierarchical U-Net)" subtitle="Binary triage screening." icon={Brain}>
                <div className="space-y-3">
                  <Metric label="Gatekeeper Prediction" value={l1?.prediction || "N/A"} tone={l1Abnormal ? "danger" : "safe"} />
                  {l1?.confidence && (
                    <Metric label="Model Confidence" value={`${Math.round(l1.confidence * 100)}%`} tone="neutral" />
                  )}
                  <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
                    <b>Explainability:</b> Diagnosis is based on a Unified Hierarchical U-Net model extracting deep structural features to triage ABNORMAL vs NORMAL scans.
                  </div>
                </div>
              </Card>

              <Card title="Level 2: Disease Router (Hierarchical U-Net)" subtitle="Specific disease classification." icon={Activity}>
                <div className="space-y-3">
                  {l1Abnormal ? (
                    <>
                      <Metric label="Router Prediction" value={l2?.prediction?.replace(/_/g, " ") || "N/A"} tone="warning" />
                      {l2?.confidence && (
                        <Metric label="Model Confidence" value={`${Math.round(l2.confidence * 100)}%`} tone="neutral" />
                      )}
                      <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
                        <b>Safety check:</b> Specific pathology successfully identified.
                      </div>
                    </>
                  ) : (
                    <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900 text-center">
                      Level 2 routing bypassed. Scan is healthy.
                    </div>
                  )}
                </div>
              </Card>

              {l1Abnormal && l3?.probs && (
                <Card title="Level 3: Multi-Morbidity Biomarkers" subtitle="Independent probabilities for co-occurring granular features." icon={Activity}>
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm font-bold text-slate-500 uppercase tracking-wide px-2">
                      <span>Biomarker</span>
                      <span className="text-right">Confidence</span>
                    </div>
                    <div className="max-h-60 overflow-y-auto pr-1 space-y-1">
                      {Object.entries(l3.probs)
                        .sort(([, a], [, b]) => b - a)
                        .map(([biomarker, prob]) => {
                          const isHigh = prob > 0.5;
                          return (
                            <div key={biomarker} className={`flex items-center justify-between rounded-xl px-3 py-2 border ${isHigh ? "bg-rose-50 border-rose-200" : "bg-slate-50 border-slate-100"}`}>
                              <span className={`font-bold ${isHigh ? "text-rose-900" : "text-slate-700"}`}>{biomarker.replace(/_/g, " ")}</span>
                              <span className={`font-semibold ${isHigh ? "text-rose-700" : "text-slate-500"}`}>{Math.round(prob * 100)}%</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </Card>
              )}

              {completed && scan.segmentation && (
                <Card title="Quantitative Clinical Metrics" subtitle="Objective, geometric measurements extracted from segmentation." icon={BarChart3}>
                  <div className="space-y-3">
                    <Metric label="Avg Retinal Thickness" value={`${Math.round(scan.segmentation.clinical_metrics.average_retinal_thickness)} px`} tone="neutral" />
                    <Metric label="Total Fluid Area" value={`${Math.round(scan.segmentation.clinical_metrics.total_fluid_area)} px²`} tone={scan.segmentation.clinical_metrics.total_fluid_area > 0 ? "warning" : "safe"} />
                    <Metric label="Max Fluid Height" value={`${Math.round(scan.segmentation.clinical_metrics.max_fluid_height)} px`} tone={scan.segmentation.clinical_metrics.max_fluid_height > 0 ? "danger" : "safe"} />
                  </div>
                </Card>
              )}

              {completed && l2?.prediction && scan.segmentation && (
                <Card title="Dynamic Clinical Interpretation" subtitle="Correlating structural findings with disease predictions." icon={BookOpen}>
                  <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-800 leading-relaxed">
                    <b>Interpretation:</b> The classifier predicts <b>{l2.prediction.replace(/_/g, " ")}</b>.{" "}
                    {scan.segmentation.clinical_metrics.total_fluid_area > 0 ? (
                      <span>This is clinically supported by the detection of <b>Intraretinal/Subretinal Fluid</b> (Area: {Math.round(scan.segmentation.clinical_metrics.total_fluid_area)} px²) in the segmented volume.</span>
                    ) : (
                      <span>No significant fluid volumes were detected in the segmented slice.</span>
                    )}
                    {scan.segmentation.clinical_metrics.average_retinal_thickness < 50 ? (
                      <span> <b>Retinal thinning</b> is observed, which may correlate with atrophic changes.</span>
                    ) : scan.segmentation.clinical_metrics.average_retinal_thickness > 150 ? (
                      <span> <b>Retinal thickening</b> is observed, consistent with edema.</span>
                    ) : null}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>

        {completed && scan?.ai_supported !== false && gradcams && Object.keys(gradcams).length > 0 && (
          <Card title="Explainability: Grad-CAM Heatmaps" subtitle="Visualizing network attention mapping across the pipeline." icon={Activity} className="lg:col-span-12 mt-5">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {gradcams.L1 && (
                <div className="space-y-2">
                  <p className="text-sm font-bold text-slate-700 text-center">Level 1 (Triage)</p>
                  <img src={gradcams.L1} alt="L1 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
                </div>
              )}
              {gradcams.L2 && (
                <div className="space-y-2">
                  <p className="text-sm font-bold text-slate-700 text-center">Level 2 (Disease Router)</p>
                  <img src={gradcams.L2} alt="L2 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
                </div>
              )}
              {gradcams.L3 && (
                <div className="space-y-2">
                  <p className="text-sm font-bold text-slate-700 text-center">Level 3 (Biomarkers)</p>
                  <img src={gradcams.L3} alt="L3 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
                </div>
              )}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
