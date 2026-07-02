export type DocCategoryColor = "blue" | "purple" | "emerald";

export interface DocArticle {
  slug: string;
  title: string;
  sidebarTitle: string;
  description: string;
  wide?: boolean;
  externalHref?: string;
}

export interface DocCategory {
  id: string;
  title: string;
  sidebarTitle: string;
  color: DocCategoryColor;
  articles: DocArticle[];
}

export const baseDocCategories: DocCategory[] = [
  {
    id: "getting-started",
    title: "Overview",
    sidebarTitle: "Overview",
    color: "blue",
    articles: [
      {
        slug: "readme",
        sidebarTitle: "Project README",
        title: "OCT/OCTA Clinical Inference Interface",
        description: "Core project overview, features, setup instructions, and architecture for the Capstone interface.",
      }
    ],
  },
  {
    id: "technical",
    title: "Technical Implementation",
    sidebarTitle: "Technical Implementation",
    color: "purple",
    articles: [
      {
        slug: "implementation-info",
        sidebarTitle: "Implementation Specs",
        title: "Implementation Details & Specs",
        description: "Technical specifications detailing integration with local servers, dicom image viewing, and python API.",
      },
      {
        slug: "ipnv2-readme",
        sidebarTitle: "IPNV2 Architecture",
        title: "IPNV2 System Architecture",
        description: "Details regarding the Inference Pipeline Network version 2 and how images are processed.",
      }
    ],
  },
  {
    id: "clinical",
    title: "Clinical Reference",
    sidebarTitle: "Clinical Reference",
    color: "emerald",
    articles: [
      {
        slug: "biomarker-mapping",
        sidebarTitle: "Biomarker Mapping",
        title: "3D OCT/OCTA Biomarker Mapping",
        description: "Layer-specific structural and vascular biomarkers with OCT/OCTA reference images and disease-feature mappings.",
        externalHref: "/public/docs/biomarker_mapping_docs/oct_biomarker_mapping.html",
      },
      {
        slug: "wireframe-demo",
        sidebarTitle: "Wireframe Demo",
        title: "Clinical Workflow Demo",
        description: "Standalone clinical workflow prototype covering triage, upload/QC, review, decision gate, and outcomes/audit screens.",
        externalHref: "/demo/",
      }
    ],
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
        externalHref: "/diagrams/?diagram=architecture",
      },
      {
        slug: "online-workflow",
        sidebarTitle: "Online Inference Workflow",
        title: "Online Clinical Inference Workflow",
        description: "Sequence diagram source for clinician upload, API ingestion, preprocessing, QC, inference, explanation, and reporting.",
        externalHref: "/diagrams/?diagram=online",
      },
      {
        slug: "offline-workflow",
        sidebarTitle: "Offline Training Workflow",
        title: "Offline Training and Validation Workflow",
        description: "Sequence diagram source for research ingestion, standardization, model training, evaluation, metrics, and versioned storage.",
        externalHref: "/diagrams/?diagram=offline",
      }
    ],
  },
];

import generatedDocs from "./modelDocsGenerated.json";

export const docCategories: DocCategory[] = [
  ...baseDocCategories,
  ...generatedDocs.categories
];

export const docSlugs = docCategories.flatMap((category) =>
  category.articles.map((article) => article.slug)
);

export function getDocArticle(slug: string): DocArticle | undefined {
  for (const category of docCategories) {
    const article = category.articles.find((entry) => entry.slug === slug);
    if (article) return article;
  }
  return undefined;
}

export const categoryColorStyles: Record<
  DocCategoryColor,
  {
    dot: string;
    hover: string;
    iconBg: string;
    iconBorder: string;
    iconText: string;
    cardFrom: string;
    cardTo: string;
    cardShadow: string;
    cardGlow: string;
    cardGlowHover: string;
    cardTitleHover: string;
    tocActive: string;
    tocShadow: string;
    cardBorderHover: string;
  }
> = {
  blue: {
    dot: "bg-cat-blue",
    hover: "hover:text-cat-blue",
    iconBg: "bg-cat-blue/10",
    iconBorder: "border-cat-blue/20",
    iconText: "text-cat-blue",
    cardFrom: "hover:from-cat-blue/10",
    cardTo: "hover:to-cat-blue/5",
    cardShadow: "hover:shadow-[0_8px_32px_-10px_var(--shadow-blue)]",
    cardGlow: "bg-cat-blue/5",
    cardGlowHover: "group-hover:bg-cat-blue/10",
    cardTitleHover: "group-hover:text-cat-blue",
    tocActive: "text-cat-blue",
    tocShadow: "bg-cat-blue shadow-[0_0_8px_var(--toc-shadow-blue)]",
    cardBorderHover: "hover:border-cat-blue",
  },
  purple: {
    dot: "bg-cat-purple",
    hover: "hover:text-cat-purple",
    iconBg: "bg-cat-purple/10",
    iconBorder: "border-cat-purple/20",
    iconText: "text-cat-purple",
    cardFrom: "hover:from-cat-purple/10",
    cardTo: "hover:to-cat-purple/5",
    cardShadow: "hover:shadow-[0_8px_32px_-10px_var(--shadow-purple)]",
    cardGlow: "bg-cat-purple/5",
    cardGlowHover: "group-hover:bg-cat-purple/10",
    cardTitleHover: "group-hover:text-cat-purple",
    tocActive: "text-cat-purple",
    tocShadow: "bg-cat-purple shadow-[0_0_8px_var(--toc-shadow-purple)]",
    cardBorderHover: "hover:border-cat-purple",
  },
  emerald: {
    dot: "bg-cat-emerald",
    hover: "hover:text-cat-emerald",
    iconBg: "bg-cat-emerald/10",
    iconBorder: "border-cat-emerald/20",
    iconText: "text-cat-emerald",
    cardFrom: "hover:from-cat-emerald/10",
    cardTo: "hover:to-cat-emerald/5",
    cardShadow: "hover:shadow-[0_8px_32px_-10px_var(--shadow-emerald)]",
    cardGlow: "bg-cat-emerald/5",
    cardGlowHover: "group-hover:bg-cat-emerald/10",
    cardTitleHover: "group-hover:text-cat-emerald",
    tocActive: "text-cat-emerald",
    tocShadow: "bg-cat-emerald shadow-[0_0_8px_var(--toc-shadow-emerald)]",
    cardBorderHover: "hover:border-cat-emerald",
  },
};
