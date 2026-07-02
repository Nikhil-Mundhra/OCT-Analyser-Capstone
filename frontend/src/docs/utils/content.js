import { marked } from 'marked';
import DOMPurify from 'dompurify';

import generatedDocs from '../modelDocsGenerated.json';

export const slugToPath = {
  'readme': '/public/docs/README.md',
  'implementation-info': '/public/docs/implementation-info.txt',
  'ipnv2-readme': '/public/docs/IPNV2_README.md',
  ...generatedDocs.slugPaths
};

export async function fetchDocContent(slug) {
  const path = slugToPath[slug];
  if (!path) return undefined;
  
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error('Failed to fetch doc');
    const text = await res.text();
    return text;
  } catch (err) {
    console.error(err);
    return undefined;
  }
}

export function extractHeadingsFromMdx(mdx) {
  const headings = [];
  const headingRegex = /^##\s+(.+)$/gm;
  let match;
  
  while ((match = headingRegex.exec(mdx)) !== null) {
    let text = match[1].trim();
    // Strip bold/italic markdown from text for ID generation
    text = text.replace(/[*_]+/g, '');
    // Strip markdown escapes (like \.)
    text = text.replace(/\\(.)/g, '$1');
    
    const id = text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const level = 2;

    let finalId = id;
    let counter = 1;
    while (headings.some(h => h.id === finalId)) {
      finalId = `${id}-${counter}`;
      counter++;
    }

    headings.push({ id: finalId, text, level });
  }

  return headings;
}

export function getHeadingIdGenerator() {
  const seenIds = new Set();
  
  return (text) => {
    let cleanText = text.replace(/[*_]+/g, '');
    const id = cleanText.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    let finalId = id;
    let counter = 1;
    while (seenIds.has(finalId)) {
      finalId = `${id}-${counter}`;
      counter++;
    }
    seenIds.add(finalId);
    return finalId;
  };
}
