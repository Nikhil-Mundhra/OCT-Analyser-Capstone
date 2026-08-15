"use client";
import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import DocLayout from '../components/DocLayout';
import DocsTableOfContents from '../components/DocsTableOfContents';
import DocArticle from '../components/DocArticle';
import { getDocArticle } from '../manifest';
import { fetchDocContent, extractHeadingsFromMdx, getHeadingIdGenerator } from '../utils/content';

export default function DocArticlePage() {
  const params = useParams();
  const slug = params?.doc;
  const [content, setContent] = useState('');
  const [headings, setHeadings] = useState([]);
  const [loading, setLoading] = useState(true);

  const articleInfo = getDocArticle(slug);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const mdx = await fetchDocContent(slug);
      if (mdx) {
        setHeadings(extractHeadingsFromMdx(mdx));
        
        // Configure marked to add IDs to headings using the same logic
        const generateId = getHeadingIdGenerator();
        const renderer = new marked.Renderer();
        renderer.heading = function({text, depth, tokens}) {
          const content = this.parser.parseInline(tokens);
          if (depth === 2) {
            const id = generateId(text);
            return `<h2 id="${id}">${content}</h2>\n`;
          }
          return `<h${depth}>${content}</h${depth}>\n`;
        };
        marked.setOptions({ renderer });
        
        const html = DOMPurify.sanitize(marked.parse(mdx));
        setContent(html);
      } else {
        setContent('<h1>Article not found</h1><p>The requested document could not be found.</p>');
        setHeadings([]);
      }
      setLoading(false);
    }
    if (slug) {
      load();
    }
  }, [slug]);

  if (!articleInfo && !loading) {
    return (
      <DocLayout>
        <div className="py-12">
          <h1 className="text-3xl font-light">Document Not Found</h1>
          <p className="mt-4 text-docs-text-secondary">We couldn't find the requested documentation page.</p>
        </div>
      </DocLayout>
    );
  }

  const toc = headings.length > 0 ? <DocsTableOfContents headings={headings} /> : null;

  return (
    <DocLayout toc={toc}>
      <DocArticle
        title={articleInfo?.title || 'Documentation'}
        description={articleInfo?.description || ''}
      >
        {articleInfo?.externalHref ? (
          <iframe 
            src={articleInfo.externalHref} 
            className="w-full h-[800px] border-0 rounded-lg shadow-sm bg-white mt-8"
            title={articleInfo.title}
          />
        ) : loading ? (
          <div className="py-8 animate-pulse">
            <div className="h-8 bg-docs-border-main rounded w-1/3 mb-6"></div>
            <div className="h-4 bg-docs-border-main rounded w-full mb-4"></div>
            <div className="h-4 bg-docs-border-main rounded w-5/6 mb-4"></div>
            <div className="h-4 bg-docs-border-main rounded w-4/6"></div>
          </div>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: content }} />
        )}
      </DocArticle>
    </DocLayout>
  );
}
