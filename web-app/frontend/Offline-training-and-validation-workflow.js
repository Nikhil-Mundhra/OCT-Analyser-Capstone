// Mermaid.js code for visualizing the offline training and validation workflow of a 3D deep learning model for OCT/OCTA analysis.
const offlineTrainingAndValidationWorkflow = String.raw`

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

    actor Researcher as AI Researcher
    participant Ingest as Data Ingestion
    participant Pre as Preprocessing
    participant QC as Quality Control
    participant Model as 3D Deep Learning Model
    participant Eval as Evaluation Framework
    participant Store as Versioned Storage

    Researcher->>Ingest: Upload OCT/OCTA scans, labels, and metadata
    Ingest->>Store: Store raw DICOM/NIfTI files
    Ingest->>Pre: Send scans for standardization

    Pre->>Pre: Parse files, normalize intensity, denoise volume
    Pre->>Pre: Resample voxel spacing and extract 3D patches
    Pre->>QC: Check scan quality and consistency

    alt Scan passes QC
        QC->>Model: Send standardized 3D tensor
        Model->>Model: Train segmentation, classification, and regression heads
        Model-->>Eval: Return masks, probabilities, thickness, and functional predictions
        Eval->>Eval: Compute Dice, AUC-ROC, sensitivity, specificity, calibration
        Eval->>Store: Save metrics, model checkpoints, logs, and failure cases
        Eval-->>Researcher: Return validation results and model comparison
    else Scan fails QC
        QC->>Store: Save failed scan and QC reason
        QC-->>Researcher: Flag scan for human audit
    end
`;

export default offlineTrainingAndValidationWorkflow;
