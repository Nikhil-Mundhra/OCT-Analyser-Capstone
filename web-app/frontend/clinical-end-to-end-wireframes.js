import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  Eye,
  FileText,
  History,
  Layers,
  Route,
  ScanLine,
  ShieldAlert,
  Stethoscope,
  Upload,
  UserRound,
} from "lucide-react";

const screens = [
  { id: "worklist", label: "1. Triage Worklist" },
  { id: "upload", label: "2. Upload and QC" },
  { id: "review", label: "3. Scan Review" },
  { id: "decision", label: "4. Human Decision Gate" },
  { id: "outcomes", label: "5. Outcomes and Safety" },
];

function StatusBadge({ children, tone = "neutral" }) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
    safe: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warning: "bg-amber-50 text-amber-800 border-amber-200",
    danger: "bg-rose-50 text-rose-700 border-rose-200",
    info: "bg-sky-50 text-sky-700 border-sky-200",
    purple: "bg-violet-50 text-violet-700 border-violet-200",
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

function Card({ title, subtitle, icon: Icon, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="mb-4 flex items-start gap-3">
        {Icon ? (
          <div className="rounded-2xl bg-slate-100 p-2 text-slate-700">
            <Icon className="h-5 w-5" />
          </div>
        ) : null}
        <div>
          <h3 className="text-base font-bold text-slate-900">{title}</h3>
          {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function WireBox({ label, className = "", height = "h-24", children }) {
  return (
    <div className={`rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 ${height} ${className}`}>
      <div className="flex h-full items-center justify-center p-4 text-center text-sm font-medium text-slate-500">
        {children || label}
      </div>
    </div>
  );
}

function Metric({ label, value, tone = "neutral" }) {
  const tones = {
    neutral: "bg-slate-50 text-slate-900 border-slate-200",
    safe: "bg-emerald-50 text-emerald-800 border-emerald-200",
    warning: "bg-amber-50 text-amber-900 border-amber-200",
    danger: "bg-rose-50 text-rose-800 border-rose-200",
    info: "bg-sky-50 text-sky-800 border-sky-200",
  };
  return (
    <div className={`rounded-2xl border p-4 ${tones[tone]}`}>
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{label}</p>
      <p className="mt-2 text-2xl font-black">{value}</p>
    </div>
  );
}

function Header() {
  return (
    <header className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-2 flex flex-wrap gap-2">
            <StatusBadge tone="info">Clinical decision support</StatusBadge>
            <StatusBadge tone="purple">Human in the loop</StatusBadge>
            <StatusBadge tone="warning">Not autonomous diagnosis</StatusBadge>
          </div>
          <h1 className="text-3xl font-black tracking-tight text-slate-950">OCT/OCTA Clinical Inference Interface</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            End-to-end clinician workflow for triage, explainable scan review, active human justification, specialist routing, and safety monitoring.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Current user</p>
            <p className="mt-1 font-bold text-slate-900">Ophthalmologist</p>
          </div>
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Mode</p>
            <p className="mt-1 font-bold text-slate-900">Review queue</p>
          </div>
        </div>
      </div>
    </header>
  );
}

function WorklistScreen() {
  const rows = [
    ["P-1029", "Macula OCT", "Low risk", "92%", "AI cleared, clinician sample review", "safe"],
    ["P-1184", "OCTA", "Ambiguous", "61%", "Send to human audit", "warning"],
    ["P-1210", "Optic disc OCT", "High risk", "88%", "Specialist review required", "danger"],
    ["P-1244", "Macula OCT", "Poor quality", "N/A", "Re-upload or manual review", "warning"],
  ];

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Triage Queue" subtitle="AI rapidly processes routine scans and protects specialist bandwidth." icon={Route} className="lg:col-span-8">
        <div className="overflow-hidden rounded-2xl border border-slate-200">
          <div className="grid grid-cols-5 bg-slate-100 px-4 py-3 text-xs font-bold uppercase tracking-wide text-slate-500">
            <span>Patient</span>
            <span>Scan</span>
            <span>AI route</span>
            <span>Confidence</span>
            <span>Action</span>
          </div>
          {rows.map((row) => (
            <div key={row[0]} className="grid grid-cols-5 items-center border-t border-slate-100 px-4 py-4 text-sm text-slate-700">
              <span className="font-bold text-slate-900">{row[0]}</span>
              <span>{row[1]}</span>
              <span><StatusBadge tone={row[5]}>{row[2]}</StatusBadge></span>
              <span className="font-bold">{row[3]}</span>
              <span>{row[4]}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Triage Rules" subtitle="Routing logic is visible to reduce blind trust." icon={ShieldAlert} className="lg:col-span-4">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-emerald-50 p-4"><b>Low risk:</b> routine queue, optional clinician sampling.</div>
          <div className="rounded-2xl bg-amber-50 p-4"><b>Ambiguous:</b> requires human audit before report sign-off.</div>
          <div className="rounded-2xl bg-rose-50 p-4"><b>High risk:</b> route to specialist with anomaly overlays and history.</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Poor QC:</b> no model claim, request re-upload or manual inspection.</div>
        </div>
      </Card>
    </div>
  );
}

function UploadScreen() {
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Scan Intake" subtitle="DICOM, NIfTI, or device export file with patient metadata." icon={Upload} className="lg:col-span-4">
        <WireBox height="h-52">
          Drag OCT/OCTA volume here<br />or select scan from imaging system
        </WireBox>
        <div className="mt-4 grid gap-3 text-sm">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Patient:</b> P-1210, Left eye</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Protocol:</b> Optic disc volume</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>History:</b> Prior optic neuritis evaluation, contrast sensitivity test attached</div>
        </div>
      </Card>

      <Card title="Quality Control Gate" subtitle="The model can be blocked before inference if scan quality is unsafe." icon={CheckCircle2} className="lg:col-span-4">
        <div className="space-y-3">
          <Metric label="Signal quality" value="Good" tone="safe" />
          <Metric label="Motion artifact" value="Mild" tone="warning" />
          <Metric label="Layer visibility" value="Accept" tone="safe" />
          <Metric label="QC decision" value="Proceed" tone="info" />
        </div>
      </Card>

      <Card title="Pipeline Status" subtitle="Clinician sees where the case is in the system." icon={Activity} className="lg:col-span-4">
        <div className="space-y-4">
          {["Upload received", "Preprocessing complete", "QC passed", "Inference running", "Report building"].map((step, index) => (
            <div key={step} className="flex items-center gap-3 rounded-2xl bg-slate-50 p-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-sm font-black text-slate-700">{index + 1}</div>
              <span className="font-semibold text-slate-700">{step}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function ReviewScreen() {
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Volumetric Scan Viewer" subtitle="Slice viewer with overlays, layer boundaries, and uncertainty map toggles." icon={ScanLine} className="lg:col-span-7">
        <div className="grid gap-4 lg:grid-cols-3">
          <WireBox height="h-72" className="lg:col-span-2">
            Main B-scan viewer<br />OCT slice with segmentation boundary overlay
          </WireBox>
          <div className="space-y-4">
            <WireBox height="h-32">En face thickness map</WireBox>
            <WireBox height="h-32">3D stack thumbnail</WireBox>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-4 gap-3 text-xs font-semibold text-slate-600">
          <button className="rounded-xl border bg-white px-3 py-2">Raw</button>
          <button className="rounded-xl border bg-white px-3 py-2">Segmentation</button>
          <button className="rounded-xl border bg-white px-3 py-2">Grad-CAM</button>
          <button className="rounded-xl border bg-white px-3 py-2">Uncertainty</button>
        </div>
      </Card>

      <Card title="AI Findings" subtitle="No hidden single answer, show confidence and competing evidence." icon={Brain} className="lg:col-span-5">
        <div className="space-y-3">
          <Metric label="Risk tier" value="High" tone="danger" />
          <Metric label="Model confidence" value="88%" tone="warning" />
          <Metric label="Calibration warning" value="Moderate" tone="warning" />
          <div className="rounded-2xl border border-rose-100 bg-rose-50 p-4 text-sm text-rose-900">
            <b>Highlighted anomaly:</b> focal thinning in temporal RNFL sector with low-confidence boundary near shadow artifact.
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
            <b>Explainability:</b> saliency concentrated around temporal nerve fiber layer, not background tissue.
          </div>
        </div>
      </Card>
    </div>
  );
}

function DecisionScreen() {
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Specialist Routing Packet" subtitle="High-risk or ambiguous scans are routed with history and highlighted anomalies." icon={Stethoscope} className="lg:col-span-5">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Recipient:</b> Neuro-ophthalmology review queue</div>
          <div className="rounded-2xl bg-rose-50 p-4"><b>Reason:</b> High-risk AI tier with temporal RNFL thinning</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Attached:</b> OCT volume, segmentation mask, heatmaps, prior visits, contrast sensitivity result</div>
          <WireBox height="h-40">Anomaly snapshot summary</WireBox>
        </div>
      </Card>

      <Card title="Active Human Justification" subtitle="Critical decisions cannot be accepted with one passive click." icon={ClipboardCheck} className="lg:col-span-7">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 p-4">
            <p className="mb-3 text-sm font-bold text-slate-900">Clinician decision</p>
            <div className="space-y-2 text-sm text-slate-700">
              <label className="block rounded-xl border p-3"><input type="radio" name="decision" className="mr-2" /> Agree with AI triage</label>
              <label className="block rounded-xl border p-3"><input type="radio" name="decision" className="mr-2" /> Override AI triage</label>
              <label className="block rounded-xl border p-3"><input type="radio" name="decision" className="mr-2" /> Defer to specialist</label>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 p-4">
            <p className="mb-3 text-sm font-bold text-slate-900">Required justification</p>
            <div className="h-40 rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
              Type rationale: image evidence, clinical history, uncertainty concerns, or reason for override.
            </div>
          </div>
        </div>
        <div className="mt-4 rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">
          <b>Automation bias guardrail:</b> Sign-off remains disabled until the clinician reviews overlays, sees confidence, and enters a decision rationale.
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white">Submit reviewed decision</button>
          <button className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">Request second opinion</button>
          <button className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">Save draft report</button>
        </div>
      </Card>
    </div>
  );
}

function OutcomesScreen() {
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Outcome and Safety Monitor" subtitle="The system is evaluated by patient impact, not isolated algorithm accuracy." icon={BarChart3} className="lg:col-span-7">
        <div className="grid gap-4 md:grid-cols-2">
          <Metric label="Time to specialist review" value="Down 28%" tone="safe" />
          <Metric label="Missed high-risk reviews" value="0 flagged" tone="safe" />
          <Metric label="Clinician overrides" value="14%" tone="info" />
          <Metric label="Adverse safety signals" value="Monitor" tone="warning" />
        </div>
        <WireBox height="h-56" className="mt-4">
          Trend chart area<br />patient outcomes, review latency, override rate, subgroup safety
        </WireBox>
      </Card>

      <Card title="Audit Trail" subtitle="Every AI claim, human action, and model version is traceable." icon={History} className="lg:col-span-5">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Model:</b> 3D-OCT-v0.8.2, calibration set May 2026</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Case events:</b> upload, QC pass, inference, explanation, clinician sign-off</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Feedback loop:</b> override cases sent to failure case repository</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Report export:</b> structured PDF and JSON summary</div>
        </div>
      </Card>
    </div>
  );
}

export default function ClinicalInterfaceWireframes() {
  const [active, setActive] = useState("worklist");
  const activeScreen = useMemo(() => screens.find((screen) => screen.id === active), [active]);

  return (
    <main className="min-h-screen bg-slate-50 p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <Header />

        <nav className="flex flex-wrap gap-2 rounded-3xl border border-slate-200 bg-white p-3 shadow-sm">
          {screens.map((screen) => (
            <button
              key={screen.id}
              onClick={() => setActive(screen.id)}
              className={`rounded-2xl px-4 py-3 text-sm font-bold transition ${
                active === screen.id
                  ? "bg-slate-900 text-white shadow-sm"
                  : "bg-slate-50 text-slate-600 hover:bg-slate-100"
              }`}
            >
              {screen.label}
            </button>
          ))}
        </nav>

        <section className="rounded-3xl border border-slate-200 bg-white/60 p-4 shadow-sm">
          <div className="mb-4 flex flex-col justify-between gap-3 lg:flex-row lg:items-center">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Wireframe screen</p>
              <h2 className="mt-1 text-2xl font-black text-slate-950">{activeScreen?.label}</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusBadge tone="info"><Clock className="mr-1 h-3 w-3" /> Fast response</StatusBadge>
              <StatusBadge tone="warning"><AlertTriangle className="mr-1 h-3 w-3" /> Confidence shown</StatusBadge>
              <StatusBadge tone="purple"><UserRound className="mr-1 h-3 w-3" /> Human justification</StatusBadge>
              <StatusBadge tone="safe"><Eye className="mr-1 h-3 w-3" /> Visual evidence</StatusBadge>
            </div>
          </div>

          {active === "worklist" && <WorklistScreen />}
          {active === "upload" && <UploadScreen />}
          {active === "review" && <ReviewScreen />}
          {active === "decision" && <DecisionScreen />}
          {active === "outcomes" && <OutcomesScreen />}
        </section>

        <footer className="grid gap-4 rounded-3xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm lg:grid-cols-4">
          <div><b className="text-slate-900">Triage:</b> low-risk cases move quickly, uncertain cases are escalated.</div>
          <div><b className="text-slate-900">Transparency:</b> confidence, uncertainty, and overlays are visible.</div>
          <div><b className="text-slate-900">Safety:</b> critical sign-off requires human rationale.</div>
          <div><b className="text-slate-900">Evaluation:</b> monitor outcomes, latency, overrides, and safety signals.</div>
        </footer>
      </div>
    </main>
  );
}

const rootElement = document.getElementById("root");

if (rootElement) {
  createRoot(rootElement).render(<ClinicalInterfaceWireframes />);
}
