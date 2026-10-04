const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_DIR = path.join(ROOT_DIR, 'web-app', 'frontend');
const FRONTEND_PUBLIC_DOCS = path.join(FRONTEND_DIR, 'public', 'docs_content');
const FRONTEND_SRC_DOCS = path.join(FRONTEND_DIR, 'src', 'docs');

function ensureDirSync(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function copyDirRecursiveSync(srcDir, destDir) {
  if (!fs.existsSync(srcDir)) return;
  ensureDirSync(destDir);
  const entries = fs.readdirSync(srcDir, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursiveSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function extractTitle(content, filename) {
  const match = content.match(/^#\s+(.+)$/m);
  if (match) {
    return match[1].replace(/<[^>]+>/g, '').trim();
  }
  return filename
    .replace('.md', '')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function extractDescription(content, fallback = '') {
  const lines = content.split('\n');
  let inNote = false;
  let description = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('#') || line.startsWith('<style') || line.startsWith('</style') || line.startsWith('<script') || line.startsWith('```')) {
      continue;
    }
    if (line === '') continue;

    if (line.startsWith('>')) {
      const cleaned = line.replace(/^>\s*(\[!.*?\])?/i, '').replace(/<[^>]+>/g, '').trim();
      if (cleaned) description.push(cleaned);
      inNote = true;
    } else if (inNote) {
      break;
    } else if (line.match(/^##\s/)) {
      continue;
    } else if (description.length === 0) {
      const cleanLine = line.replace(/<[^>]+>/g, '').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').trim();
      if (cleanLine.length > 20) {
        description.push(cleanLine);
        break;
      }
    }
  }

  const text = description.join(' ').substring(0, 160).trim();
  if (text) {
    return text.length >= 160 ? text + '...' : text;
  }
  return fallback;
}

// Master Article Definitions
const CATEGORY_DEFINITIONS = [
  {
    id: "getting-started",
    title: "Overview & System Architecture",
    sidebarTitle: "Overview",
    color: "blue",
    articles: [
      {
        slug: "readme",
        title: "OCT/OCTA Clinical Inference Interface",
        sidebarTitle: "Project Overview",
        description: "Core platform overview, multi-model segmentation pipeline, clinical workflows, and setup guide.",
        sourceFile: path.join(ROOT_DIR, 'README.md'),
        publicPath: "/docs_content/README.md"
      },
      {
        slug: "docs-catalog",
        title: "Clinical Documentation Catalog",
        sidebarTitle: "Documentation Index",
        description: "Comprehensive catalog of clinical cohort benchmarks, anatomical references, and technical specifications.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'README.md'),
        publicPath: "/docs_content/docs_catalog.md"
      }
    ]
  },
  {
    id: "cohort-reports",
    title: "Clinical Cohort Benchmark Reports",
    sidebarTitle: "Cohort Reports",
    color: "emerald",
    articles: [
      {
        slug: "cohort-3d-vs-biplanar-comparison",
        title: "3D Volumetric vs. Bi-Planar Orthogonal Benchmark",
        sidebarTitle: "3D vs. Bi-Planar Study",
        description: "Comparative study benchmarking 3D anisotropic U-Net against bi-planar orthogonal 2.5D fusion across 46 Solix volumes.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '3d_vs_biplanar_comparison'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '3d_vs_biplanar_comparison', 'research_report_3d_vs_biplanar_comparison.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '3d_vs_biplanar_comparison', 'research_report_3d_vs_biplanar_comparison.pdf'),
        destSubdir: "cohort_reports/3d_vs_biplanar_comparison",
        destFilename: "research_report_3d_vs_biplanar_comparison.md",
        pdfDestFilename: "research_report_3d_vs_biplanar_comparison.pdf"
      },
      {
        slug: "cohort-2026-10-04-rnfl-3d-expanded",
        title: "Dense Anisotropic 3D U-Net Benchmark (Job 18574378)",
        sidebarTitle: "3D U-Net (Job 18574378)",
        description: "Executive clinical evaluation of the full volumetric 3D U-Net across the 23-subject expanded Solix cohort.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-04_rnfl_3d_expanded_18574378'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-04_rnfl_3d_expanded_18574378', 'executive_cohort_rnfl_report_3d_job_18574378.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-04_rnfl_3d_expanded_18574378', 'executive_cohort_rnfl_report_3d_job_18574378.pdf'),
        destSubdir: "cohort_reports/2026-10-04_rnfl_3d_expanded_18574378",
        destFilename: "executive_cohort_rnfl_report_3d_job_18574378.md",
        pdfDestFilename: "executive_cohort_rnfl_report_3d_job_18574378.pdf"
      },
      {
        slug: "cohort-2026-10-03-biplanar-expanded",
        title: "Bi-Planar Orthogonal Expanded Cohort (Job 18563914)",
        sidebarTitle: "Bi-Planar (Job 18563914)",
        description: "Automated cohort benchmark of the bi-planar orthogonal architecture with corrected OS coordinate restoration.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-03_biplanar_expanded_18563914'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-03_biplanar_expanded_18563914', 'executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-10-03_biplanar_expanded_18563914', 'executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.pdf'),
        destSubdir: "cohort_reports/2026-10-03_biplanar_expanded_18563914",
        destFilename: "executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.md",
        pdfDestFilename: "executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.pdf"
      },
      {
        slug: "cohort-multi-model-validation",
        title: "Multi-Model Validation Benchmark Report",
        sidebarTitle: "Multi-Model Benchmark",
        description: "Cross-model validation comparing 2D U-Net, 2.5D ResNet, Bi-Planar Orthogonal, and 3D U-Net architectures.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'multi_model_validation_benchmark'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'multi_model_validation_benchmark', 'multi_model_validation_benchmark_report.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'multi_model_validation_benchmark', 'multi_model_validation_benchmark_report.pdf'),
        destSubdir: "cohort_reports/multi_model_validation_benchmark",
        destFilename: "multi_model_validation_benchmark_report.md",
        pdfDestFilename: "multi_model_validation_benchmark_report.pdf"
      },
      {
        slug: "cohort-2026-09-28-biplanar-expanded",
        title: "Bi-Planar Orthogonal Expanded Benchmark (Job 18266075)",
        sidebarTitle: "Bi-Planar (Job 18266075)",
        description: "Expanded cohort benchmark of bi-planar orthogonal model on held-out validation eyes.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-28_biplanar_expanded_18266075'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-28_biplanar_expanded_18266075', 'executive_cohort_rnfl_report_biplanar.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-28_biplanar_expanded_18266075', 'executive_cohort_rnfl_report_biplanar.pdf'),
        destSubdir: "cohort_reports/2026-09-28_biplanar_expanded_18266075",
        destFilename: "executive_cohort_rnfl_report_biplanar.md",
        pdfDestFilename: "executive_cohort_rnfl_report_biplanar.pdf"
      },
      {
        slug: "cohort-2026-09-26-biplanar-orthogonal",
        title: "Bi-Planar Orthogonal Heavy Model (Job 18223981 - SOTA)",
        sidebarTitle: "Bi-Planar SOTA (Job 18223981)",
        description: "Landmark SOTA evaluation achieving 3.02 µm OD MABE and 4.62 µm validation MABE with dual-view orthogonal fusion.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-26_biplanar_orthogonal_18223981'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-26_biplanar_orthogonal_18223981', 'executive_cohort_rnfl_report_biplanar.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-26_biplanar_orthogonal_18223981', 'executive_cohort_rnfl_report_biplanar.pdf'),
        destSubdir: "cohort_reports/2026-09-26_biplanar_orthogonal_18223981",
        destFilename: "executive_cohort_rnfl_report_biplanar.md",
        pdfDestFilename: "executive_cohort_rnfl_report_biplanar.pdf"
      },
      {
        slug: "cohort-reports-index",
        title: "Clinical Cohort Reports Index & Synthesis",
        sidebarTitle: "Cohort Reports Index",
        description: "Executive summary and comparative matrix across all historical clinical segmentation runs.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'README.md'),
        publicPath: "/docs_content/cohort_reports/README.md"
      },
      {
        slug: "cohort-2026-09-21-heavy-volumetric",
        title: "Volumetric 2.5D ResNet Heavy (Job 18045386)",
        sidebarTitle: "Volumetric Heavy (18045386)",
        description: "Evaluation of the 6.58M-parameter volumetric 2.5D ResNet architecture across 23 subjects.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_heavy_volumetric_18045386'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_heavy_volumetric_18045386', 'executive_cohort_rnfl_report_heavy_18045386.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_heavy_volumetric_18045386', 'executive_cohort_rnfl_report_heavy_18045386.pdf'),
        destSubdir: "cohort_reports/2026-09-21_heavy_volumetric_18045386",
        destFilename: "executive_cohort_rnfl_report_heavy_18045386.md",
        pdfDestFilename: "executive_cohort_rnfl_report_heavy_18045386.pdf"
      },
      {
        slug: "cohort-2026-09-21-light-volumetric",
        title: "Volumetric 2.5D ResNet Light (Job 18043443)",
        sidebarTitle: "Volumetric Light (18043443)",
        description: "Compact 1.65M-parameter 2.5D ResNet model evaluation on the Solix cohort.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_light_volumetric_18043443'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_light_volumetric_18043443', 'executive_cohort_rnfl_report_light_18043443.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_light_volumetric_18043443', 'executive_cohort_rnfl_report_light_18043443.pdf'),
        destSubdir: "cohort_reports/2026-09-21_light_volumetric_18043443",
        destFilename: "executive_cohort_rnfl_report_light_18043443.md",
        pdfDestFilename: "executive_cohort_rnfl_report_light_18043443.pdf"
      },
      {
        slug: "cohort-2026-09-21-expanded-23subj",
        title: "Expanded 23-Subject Solix Baseline",
        sidebarTitle: "23-Subject Solix Baseline",
        description: "Baseline benchmark evaluating initial commercial segmentation heuristic against clinician corrections.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_expanded_cohort_23subj'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_expanded_cohort_23subj', 'executive_cohort_rnfl_report_23subj_20260921.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-21_expanded_cohort_23subj', 'executive_cohort_rnfl_report_23subj_20260921.pdf'),
        destSubdir: "cohort_reports/2026-09-21_expanded_cohort_23subj",
        destFilename: "executive_cohort_rnfl_report_23subj_20260921.md",
        pdfDestFilename: "executive_cohort_rnfl_report_23subj_20260921.pdf"
      },
      {
        slug: "cohort-2026-09-14-pilot-11subj",
        title: "Pilot 11-Subject Benchmark",
        sidebarTitle: "Pilot 11-Subject Cohort",
        description: "Preliminary feasibility evaluation across 11 human subjects (22 eye volumes).",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-14_pilot_cohort_11subj'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-14_pilot_cohort_11subj', 'executive_cohort_rnfl_report_11subj_20260914.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', '2026-09-14_pilot_cohort_11subj', 'executive_cohort_rnfl_report_11subj_20260914.pdf'),
        destSubdir: "cohort_reports/2026-09-14_pilot_cohort_11subj",
        destFilename: "executive_cohort_rnfl_report_11subj_20260914.md",
        pdfDestFilename: "executive_cohort_rnfl_report_11subj_20260914.pdf"
      },
      {
        slug: "cohort-baseline-unet-vs-heuristic",
        title: "Commercial Solix Algorithm vs. 2D U-Net",
        sidebarTitle: "2D U-Net vs Solix",
        description: "Foundational comparison of standard 2D slice-wise U-Net against the Optovue commercial heuristic.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'baseline_unet_vs_heuristic_report'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'baseline_unet_vs_heuristic_report', 'unet_vs_algorithm_segmentation_report.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'baseline_unet_vs_heuristic_report', 'unet_vs_algorithm_segmentation_report.pdf'),
        destSubdir: "cohort_reports/baseline_unet_vs_heuristic_report",
        destFilename: "unet_vs_algorithm_segmentation_report.md",
        pdfDestFilename: "unet_vs_algorithm_segmentation_report.pdf"
      },
      {
        slug: "cohort-consolidated-archive",
        title: "Consolidated Milestone Executive Archive",
        sidebarTitle: "Consolidated Archive",
        description: "Archived executive synthesis of early milestone results and clinical audit findings.",
        sourceDir: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'consolidated_executive_archive'),
        sourceFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'consolidated_executive_archive', 'executive_cohort_rnfl_report_final.md'),
        pdfFile: path.join(ROOT_DIR, 'docs', 'cohort_reports', 'consolidated_executive_archive', 'executive_cohort_rnfl_report_final.pdf'),
        destSubdir: "cohort_reports/consolidated_executive_archive",
        destFilename: "executive_cohort_rnfl_report_final.md",
        pdfDestFilename: "executive_cohort_rnfl_report_final.pdf"
      }
    ]
  },
  {
    id: "clinical",
    title: "Clinical Reference Guides",
    sidebarTitle: "Clinical Guides",
    color: "emerald",
    articles: [
      {
        slug: "retinal-layers-guide",
        title: "Retinal Layer Boundaries & Optic Neuritis OCT Guide",
        sidebarTitle: "Retinal Layers Guide",
        description: "Detailed anatomical guide covering all 11 retinal interfaces, thickness profiles, and neuro-ophthalmic pathology patterns.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'clinical_guides', 'retinal_layers_optic_neuritis_oct_guide.md'),
        publicPath: "/docs_content/clinical_guides/retinal_layers_optic_neuritis_oct_guide.md"
      },
      {
        slug: "biomarker-mapping",
        title: "3D OCT/OCTA Biomarker Mapping",
        sidebarTitle: "Biomarker Mapping",
        description: "Layer-specific structural and vascular biomarkers with OCT/OCTA reference images and disease-feature mappings.",
        externalHref: "/docs_content/biomarker_mapping_docs/oct_biomarker_mapping.html"
      },
      {
        slug: "wireframe-demo",
        title: "Clinical Workflow Demo",
        sidebarTitle: "Wireframe Demo",
        description: "Standalone clinical workflow prototype covering triage, upload/QC, review, decision gate, and outcomes/audit screens.",
        externalHref: "/demo/"
      }
    ]
  },
  {
    id: "technical",
    title: "Technical Implementation & Specifications",
    sidebarTitle: "Technical Specs",
    color: "purple",
    articles: [
      {
        slug: "implementation-info",
        title: "Implementation Details & Architectural Specs",
        sidebarTitle: "Implementation Specs",
        description: "Technical specifications detailing integration with local servers, DICOM volume ingestion, and Python API.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'implementation_info.md'),
        publicPath: "/docs_content/technical_specs/implementation_info.md"
      },
      {
        slug: "hierarchical-classification-architecture",
        title: "Hierarchical Multi-Stage Classification Architecture",
        sidebarTitle: "Classification Architecture",
        description: "Architectural blueprint for the multi-head hierarchical pathology classifier and triage decision engine.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'hierarchical_classification_architecture.md'),
        publicPath: "/docs_content/technical_specs/hierarchical_classification_architecture.md"
      },
      {
        slug: "preprocessing-tuning-guide",
        title: "Spatial Geometry Normalization & Preprocessing Tuning Guide",
        sidebarTitle: "Preprocessing Guide",
        description: "Guidelines for white-bar masking, bounding-box tissue cropping, aspect ratio preservation, and intensity normalization.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'preprocessing_tuning_guide.md'),
        publicPath: "/docs_content/technical_specs/preprocessing_tuning_guide.md"
      },
      {
        slug: "oct-segmentation-dataset-search",
        title: "Systematic Survey of Public & Clinical OCT Datasets",
        sidebarTitle: "Dataset Survey",
        description: "Comprehensive evaluation of DUKE, RETOUCH, AROI, and Solix clinical repositories for deep-learning transferability.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'oct_segmentation_dataset_search.md'),
        publicPath: "/docs_content/technical_specs/oct_segmentation_dataset_search.md"
      },
      {
        slug: "oct-layer-segmentation-validation",
        title: "OCT Layer Segmentation Validation & Clinician Correction Release",
        sidebarTitle: "Validation Protocols",
        description: "Quality control protocols, expert audit methodology, and manual curve correction procedures.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'oct_layer_segmentation_validation_correction_release.md'),
        publicPath: "/docs_content/technical_specs/oct_layer_segmentation_validation_correction_release.md"
      },
      {
        slug: "sam-medsam-postmortem",
        title: "Postmortem: SAM & MedSAM Zero-Shot Retinal Segmentation Failure",
        sidebarTitle: "SAM/MedSAM Postmortem",
        description: "Detailed postmortem analyzing why foundational vision models fail at fine retinal layer boundary extraction.",
        sourceFile: path.join(ROOT_DIR, 'docs', 'technical_specs', 'postmortems', 'SAM_MEDSAM_RETINAL_SEGMENTATION_FAILURE.md'),
        publicPath: "/docs_content/technical_specs/postmortems/SAM_MEDSAM_RETINAL_SEGMENTATION_FAILURE.md"
      }
    ]
  },
  {
    id: "classification",
    title: "Classification Models & Pipelines",
    sidebarTitle: "Classification",
    color: "purple",
    articles: [
      {
        slug: "classification-readme",
        title: "Classification Models Overview",
        sidebarTitle: "Overview",
        description: "Multi-head ConvNeXt models for ONH and retinal pathology categorization.",
        sourceFile: path.join(ROOT_DIR, 'training', 'classification', 'Documentation', 'README.md'),
        publicPath: "/docs_content/models/classification/README.md"
      },
      {
        slug: "classification-architecture",
        title: "Multi-Head ConvNeXt Architecture",
        sidebarTitle: "Architecture",
        description: "Detailed architecture of the multi-head pathology classifier.",
        sourceFile: path.join(ROOT_DIR, 'training', 'classification', 'Documentation', 'architecture.md'),
        publicPath: "/docs_content/models/classification/architecture.md"
      },
      {
        slug: "classification-data-augmentation",
        title: "Classification Data Augmentation",
        sidebarTitle: "Data Augmentation",
        description: "Domain-specific perturbations and augmentations for ophthalmic OCT scans.",
        sourceFile: path.join(ROOT_DIR, 'training', 'classification', 'Documentation', 'data_augmentation.md'),
        publicPath: "/docs_content/models/classification/data_augmentation.md"
      },
      {
        slug: "classification-training-pipeline",
        title: "Classification Training Pipeline",
        sidebarTitle: "Training Pipeline",
        description: "Training loops, loss weighting, and scheduler configurations for the hierarchical classifier.",
        sourceFile: path.join(ROOT_DIR, 'training', 'classification', 'Documentation', 'training_pipeline.md'),
        publicPath: "/docs_content/models/classification/training_pipeline.md"
      }
    ]
  },
  {
    id: "diagrams",
    title: "Architecture & Workflows",
    sidebarTitle: "Architecture Diagrams",
    color: "blue",
    articles: [
      {
        slug: "architecture-flowchart",
        sidebarTitle: "Architecture Flowchart",
        title: "Deep Learning Architecture Flowchart",
        description: "Mermaid source for the 3D tensor pipeline, shared backbone, prediction heads, uncertainty, and report assembly.",
        externalHref: "/diagrams/?diagram=architecture"
      },
      {
        slug: "online-workflow",
        sidebarTitle: "Online Inference Workflow",
        title: "Online Clinical Inference Workflow",
        description: "Sequence diagram source for clinician upload, API ingestion, preprocessing, QC, inference, explanation, and reporting.",
        externalHref: "/diagrams/?diagram=online"
      },
      {
        slug: "offline-workflow",
        sidebarTitle: "Offline Training Workflow",
        title: "Offline Training and Validation Workflow",
        description: "Sequence diagram source for research ingestion, standardization, model training, evaluation, metrics, and versioned storage.",
        externalHref: "/diagrams/?diagram=offline"
      }
    ]
  }
];

function processDocumentationSync() {
  console.log("Starting Capstone documentation synchronization...");

  // Ensure base destination directories exist
  ensureDirSync(FRONTEND_PUBLIC_DOCS);
  ensureDirSync(FRONTEND_SRC_DOCS);

  // Copy clinical guides assets (biomarker mapping HTML & images)
  const biomarkerSrc = path.join(ROOT_DIR, 'docs', 'clinical_guides', 'biomarker_mapping');
  const biomarkerDest = path.join(FRONTEND_PUBLIC_DOCS, 'biomarker_mapping_docs');
  if (fs.existsSync(biomarkerSrc)) {
    copyDirRecursiveSync(biomarkerSrc, biomarkerDest);
  }

  // Copy technical specs assets if present
  const techAssetsSrc = path.join(ROOT_DIR, 'docs', 'technical_specs', 'assets');
  const techAssetsDest = path.join(FRONTEND_PUBLIC_DOCS, 'technical_specs', 'assets');
  if (fs.existsSync(techAssetsSrc)) {
    copyDirRecursiveSync(techAssetsSrc, techAssetsDest);
  }

  // Build a lookup map of filenames/basenames to slugs for link rewriting
  const filenameToSlug = {};
  const slugToPublicPath = {};
  const outputCategories = [];

  // Register all articles in lookup tables
  CATEGORY_DEFINITIONS.forEach(category => {
    category.articles.forEach(article => {
      if (article.sourceFile) {
        const basename = path.basename(article.sourceFile);
        filenameToSlug[basename] = article.slug;
        const relToDocs = path.relative(path.join(ROOT_DIR, 'docs'), article.sourceFile).replace(/\\/g, '/');
        filenameToSlug[relToDocs] = article.slug;
      }
    });
  });

  // Process and copy all articles
  CATEGORY_DEFINITIONS.forEach(category => {
    const articles = [];

    category.articles.forEach(article => {
      if (article.externalHref) {
        // External interactive article
        articles.push({
          slug: article.slug,
          title: article.title,
          sidebarTitle: article.sidebarTitle || article.title,
          description: article.description,
          externalHref: article.externalHref
        });
        return;
      }

      if (!article.sourceFile || !fs.existsSync(article.sourceFile)) {
        console.warn(`Warning: Source file not found for slug '${article.slug}': ${article.sourceFile}`);
        return;
      }

      // Determine destination paths
      let destFilePath = "";
      let publicDocUrl = "";
      let pdfPublicUrl = undefined;

      if (article.destSubdir) {
        const destDir = path.join(FRONTEND_PUBLIC_DOCS, article.destSubdir);
        ensureDirSync(destDir);
        destFilePath = path.join(destDir, article.destFilename || path.basename(article.sourceFile));
        publicDocUrl = `/docs_content/${article.destSubdir}/${article.destFilename || path.basename(article.sourceFile)}`;

        // Copy directory assets if sourceDir has assets/
        if (article.sourceDir) {
          const assetsDir = path.join(article.sourceDir, 'assets');
          if (fs.existsSync(assetsDir)) {
            const destAssetsDir = path.join(destDir, 'assets');
            copyDirRecursiveSync(assetsDir, destAssetsDir);
          }
        }

        // Copy PDF if exists
        if (article.pdfFile && fs.existsSync(article.pdfFile)) {
          const pdfDestPath = path.join(destDir, article.pdfDestFilename || path.basename(article.pdfFile));
          fs.copyFileSync(article.pdfFile, pdfDestPath);
          pdfPublicUrl = `/docs_content/${article.destSubdir}/${article.pdfDestFilename || path.basename(article.pdfFile)}`;
        }
      } else if (article.publicPath) {
        destFilePath = path.join(FRONTEND_DIR, 'public', article.publicPath.replace(/^\//, ''));
        ensureDirSync(path.dirname(destFilePath));
        publicDocUrl = article.publicPath;
      }

      // Read markdown and perform transformations
      let content = fs.readFileSync(article.sourceFile, 'utf-8');

      // Rewrite image links relative to assets
      const assetPrefix = article.destSubdir ? `/docs_content/${article.destSubdir}/assets` : `/docs_content/assets`;

      // 1. Markdown images: ![alt](assets/...) or ![alt](./assets/...)
      content = content.replace(/!\[(.*?)\]\((?:\.\/)?assets\/(.*?)\)/g, `![$1](${assetPrefix}/$2)`);
      // 2. HTML <img> tags: <img ... src="assets/..." ...>
      content = content.replace(/<img(.*?)src=(["'])(?:\.\/)?assets\/(.*?)\2(.*?)>/g, `<img$1src=$2${assetPrefix}/$3$2$4>`);

      // 3. Rewrite relative PDF links
      if (article.destSubdir) {
        const docSubdirPrefix = `/docs_content/${article.destSubdir}`;
        content = content.replace(/\]\((?:\.\/)?([a-zA-Z0-9_-]+\.pdf)\)/g, `](${docSubdirPrefix}/$1)`);
        content = content.replace(/href=(["'])(?:\.\/)?([a-zA-Z0-9_-]+\.pdf)\1/g, `href=$1${docSubdirPrefix}/$2$1`);
      }

      // 4. Rewrite markdown links to mapped slugs
      content = content.replace(/\]\((?!http|\/)([^)]+)\.md(#.*)?\)/g, (match, relPath, hash) => {
        const cleanRelPath = relPath.replace(/^\.\//, '');
        const targetBasename = path.basename(cleanRelPath) + '.md';

        if (filenameToSlug[targetBasename]) {
          return `](/docs/${filenameToSlug[targetBasename]}${hash || ''})`;
        }
        if (filenameToSlug[cleanRelPath]) {
          return `](/docs/${filenameToSlug[cleanRelPath]}${hash || ''})`;
        }
        return match;
      });

      // Write transformed content to destination
      fs.writeFileSync(destFilePath, content, 'utf-8');

      slugToPublicPath[article.slug] = publicDocUrl;

      const title = article.title || extractTitle(content, path.basename(article.sourceFile));
      const description = article.description || extractDescription(content, `Documentation for ${title}`);

      articles.push({
        slug: article.slug,
        title,
        sidebarTitle: article.sidebarTitle || title,
        description,
        pdfHref: pdfPublicUrl,
        wide: article.wide || false
      });
    });

    if (articles.length > 0) {
      outputCategories.push({
        id: category.id,
        title: category.title,
        sidebarTitle: category.sidebarTitle || category.title,
        color: category.color,
        articles
      });
    }
  });

  // Write out the generated JSON
  const outputJsonPath = path.join(FRONTEND_SRC_DOCS, 'modelDocsGenerated.json');
  fs.writeFileSync(outputJsonPath, JSON.stringify({
    categories: outputCategories,
    slugPaths: slugToPublicPath
  }, null, 2), 'utf-8');

  console.log(`Successfully synced documentation! Generated ${outputCategories.length} categories with ${Object.keys(slugToPublicPath).length} indexed documents.`);
}

processDocumentationSync();
