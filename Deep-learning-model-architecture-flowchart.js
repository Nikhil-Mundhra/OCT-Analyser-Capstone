// Mermaid.js code for visualizing the deep learning model architecture flowchart.
const deepLearningModelArchitectureFlowchart = String.raw`
%%{init: {
  "theme": "base",
  "flowchart": {
    "htmlLabels": true,
    "curve": "basis",
    "nodeSpacing": 90,
    "rankSpacing": 100,
    "padding": 25
  },
  "themeVariables": {
    "fontSize": "22px",
    "fontFamily": "Arial",
    "primaryTextColor": "#111111",
    "lineColor": "#111111",
    "tertiaryColor": "#F8FAFC"
  },
  "themeCSS": "
    .node rect, .node polygon, .node path {
      stroke-width: 3px !important;
      rx: 14px !important;
      ry: 14px !important;
    }
    .edgePath path {
      stroke-width: 3px !important;
    }
    .cluster rect {
      stroke-width: 3px !important;
      rx: 18px !important;
      ry: 18px !important;
    }
    .edgeLabel {
      font-size: 19px !important;
      font-weight: 500 !important;
    }
    text {
      font-weight: 500 !important;
    }
  "
}}%%

flowchart TB

%% Layout Rules:
%% 1. Main flow is top-to-bottom.
%% 2. Each stage is placed inside a container.
%% 3. Parallel model heads are separated into lanes.
%% 4. Avoid many-to-many arrows.
%% 5. Merge outputs only at the final report assembly node.

    subgraph C1["1. Input and Preprocessing"]
        direction TB

        A["Raw OCT / OCTA Volume<br/>DICOM, NIfTI, or .vol"]
        B["Preprocessing Pipeline<br/>Intensity clipping<br/>Denoising<br/>Voxel resampling<br/>Retinal flattening"]
        C["Standardized 3D Tensor<br/>C × D × H × W"]

        A --> B --> C
    end

    subgraph C2["2. Shared 3D Deep Learning Backbone"]
        direction TB

        D["3D U-Net / 3D CNN Backbone<br/>3D convolution blocks<br/>Residual connections<br/>Encoder-decoder structure<br/>Skip connections"]
        E["Shared Volumetric Feature Map<br/>Learns retinal structure across slices"]

        D --> E
    end

    C --> D

    subgraph C3["3. Parallel Prediction Heads"]
        direction LR

        F["Segmentation Head<br/>10+ retinal layer classes<br/>Voxel-level layer mask"]
        G["Classification Head<br/>Disease / risk probability<br/>Clinical label prediction"]
        H["Regression Head<br/>Retinal thickness prediction<br/>Contrast sensitivity prediction"]
        I["Uncertainty Head<br/>Monte Carlo dropout<br/>Pixel-level confidence"]
    end

    E --> F
    E --> G
    E --> H
    E --> I

    subgraph C4["4. Segmentation Safety and Anatomical Correction"]
        direction TB

        J["Raw Segmentation Mask"]
        K["Anatomical Constraint Layer<br/>Layer-order rules<br/>Topology preservation<br/>Non-crossing boundaries"]
        L["Graph-Search Correction<br/>Viterbi / Dijkstra-style refinement<br/>Shadow-gap interpolation"]
        M["Corrected Segmentation Mask<br/>Clinically plausible retinal layers"]

        J --> K --> L --> M
    end

    F --> J

    subgraph C5["5. Output Processing Lanes"]
        direction LR

        N["Biomarker Engine<br/>Layer thickness maps<br/>Fluid volume<br/>RPE elevation metrics"]

        O["Prediction Formatter<br/>Predicted class<br/>Risk score<br/>Confidence score"]

        P["Functional Output<br/>Contrast sensitivity estimate<br/>Visual function prediction"]

        Q["Uncertainty Visualization<br/>Low-confidence regions<br/>Artifact vs pathology support"]

        R["Explainability Engine<br/>3D Grad-CAM<br/>Saliency maps<br/>Slice-wise heatmaps"]
    end

    M --> N
    G --> O
    H --> P
    I --> Q
    E --> R

    subgraph C6["6. Final Clinical Output"]
        direction TB

        S["Report Assembly<br/>Combine predictions, biomarkers,<br/>uncertainty, and explanations"]

        T["Clinical Report / FastAPI Response<br/>JSON output<br/>Heatmap references<br/>Model version<br/>Processing time"]

        S --> T
    end

    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S

    classDef input fill:#EAF4FF,stroke:#2563EB,color:#111111;
    classDef backbone fill:#ECFDF5,stroke:#059669,color:#111111;
    classDef heads fill:#F3E8FF,stroke:#7C3AED,color:#111111;
    classDef safety fill:#FEF2F2,stroke:#DC2626,color:#111111;
    classDef processing fill:#FFF7ED,stroke:#EA580C,color:#111111;
    classDef final fill:#F8FAFC,stroke:#111827,color:#111111;

    class A,B,C input;
    class D,E backbone;
    class F,G,H,I heads;
    class J,K,L,M safety;
    class N,O,P,Q,R processing;
    class S,T final;
`;

export default deepLearningModelArchitectureFlowchart;
