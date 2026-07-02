import React from 'react';

export default function DocArticle({ title, description, children }) {
  return (
    <div className="docs-content-wrapper">
      <div className="docs-header mb-12 border-b border-docs-border-main pb-8 transition-colors duration-200">
        <h1 className="text-4xl font-light text-docs-text-primary tracking-tight mb-4 transition-colors duration-200">{title}</h1>
        <p className="text-xl text-docs-text-secondary transition-colors duration-200">{description}</p>
      </div>
      <div className="docs-content">
        {children}
      </div>
    </div>
  );
}

