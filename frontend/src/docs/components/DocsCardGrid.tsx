import { Link } from "react-router-dom";
import {
  categoryColorStyles,
  type DocCategory,
} from "../manifest";

interface DocsCardGridProps {
  category: DocCategory;
}

export default function DocsCardGrid({ category }: DocsCardGridProps) {
  const styles = categoryColorStyles[category.color];

  return (
    <section
      id={category.id}
      className="mb-24 scroll-mt-28 relative"
    >
      <div className="flex items-center gap-4 mb-8">
        <div
          className={`p-2.5 rounded-xl ${styles.iconBg} border ${styles.iconBorder} ${styles.iconText}`}
        >
          <CategoryIcon categoryId={category.id} />
        </div>
        <h2 className="text-3xl font-light text-docs-text-primary transition-colors duration-200">{category.title}</h2>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        {category.articles.map((article) => (
          <Link
            key={article.slug}
            href={`/docs/${article.slug}`}
            className={`group relative flex flex-col rounded-2xl border border-docs-card-border bg-docs-bg-card p-6 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md ${styles.cardBorderHover} ${article.wide ? "col-span-2" : ""}`}
          >
            <h3
              className={`text-lg font-light text-docs-text-primary mb-3 ${styles.cardTitleHover} transition-colors duration-200`}
            >
              {article.title}
            </h3>
            <p className="text-sm text-docs-text-muted flex-1 leading-relaxed transition-colors duration-200">
              {article.description}
            </p>
          </Link>
        ))}
      </div>
    </section>
  );
}

function CategoryIcon({ categoryId }: { categoryId: string }) {
  if (categoryId === "getting-started") {
    return (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M13 10V3L4 14h7v7l9-11h-7z"
        />
      </svg>
    );
  }

  if (categoryId === "product-info") {
    return (
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
        />
      </svg>
    );
  }

  return (
    <svg
      className="w-6 h-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
      />
    </svg>
  );
}
