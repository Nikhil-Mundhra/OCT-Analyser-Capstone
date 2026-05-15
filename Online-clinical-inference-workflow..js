// Mermaid.js code for visualizing the online clinical interface workflow of a 3D deep learning model for OCT/OCTA analysis.
const onlineClinicalInferenceWorkflow = String.raw`

%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "22px",
    "fontFamily": "Arial",
    "actorFontSize": "22px",
    "messageFontSize": "21px",
    "noteFontSize": "21px",
    "signalColor": "#111111",
    "signalTextColor": "#111111"
  },
  "themeCSS": ".messageLine0, .messageLine1 { stroke-width: 3px !important; } .actor-line { stroke-width: 3px !important; } text { font-weight: 500 !important; }"
}}%%

sequenceDiagram
    autonumber

    actor Clinician as Clinician
    participant UI as Clinical Interface
    participant API as FastAPI Service
    participant Ingest as Data Ingestion
    participant Pre as Preprocessing
    participant QC as Quality Control
    participant Model as 3D Model
    participant XAI as Explainability Engine
    participant Report as Report Builder
    participant Store as Logs and Artifacts

    Clinician->>UI: Upload OCT/OCTA scan
    UI->>API: POST /upload-scan
    API->>Ingest: Receive scan and metadata
    Ingest->>Store: Store uploaded scan
    Ingest->>Pre: Trigger preprocessing

    Pre->>Pre: Convert scan into standardized 3D tensor
    Pre->>QC: Assess scan quality

    alt Good quality scan
        QC-->>API: Approve scan for inference
        API->>Model: POST /predict

        Model->>Model: Run GPU inference
        Model-->>Report: Return predictions

        Report->>Report: Calculate retinal thickness metrics
        Report->>Report: Add disease probability and confidence score
        Report->>Report: Add contrast sensitivity prediction

        API->>XAI: POST /explain
        XAI->>XAI: Generate 3D Grad-CAM and saliency maps
        XAI-->>Report: Return slice-wise heatmap overlays

        Report->>Store: Save prediction, heatmaps, latency, and model version
        Report-->>API: Structured JSON report
        API-->>UI: Return predictions and heatmap references
        UI-->>Clinician: Display clinical report with visual explanation

    else Bad quality scan
        QC-->>API: Reject scan with QC reason
        API-->>UI: Return quality warning
        UI-->>Clinician: Ask for re-upload or review

    else Needs human audit
        QC-->>API: Flag scan for review
        API-->>UI: Return audit-required status
        UI-->>Clinician: Display human audit required message
        QC->>Store: Save audit flag and QC evidence
    end
`;

export default onlineClinicalInferenceWorkflow;
