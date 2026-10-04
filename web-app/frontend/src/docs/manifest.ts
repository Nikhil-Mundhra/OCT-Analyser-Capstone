export type DocCategoryColor = "blue" | "purple" | "emerald";

export interface DocArticle {
  slug: string;
  title: string;
  sidebarTitle: string;
  description: string;
  wide?: boolean;
  externalHref?: string;
  pdfHref?: string;
}

export interface DocCategory {
  id: string;
  title: string;
  sidebarTitle: string;
  color: DocCategoryColor;
  articles: DocArticle[];
}

import generatedDocs from "./modelDocsGenerated.json";

export const docCategories: DocCategory[] = generatedDocs.categories as DocCategory[];

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
