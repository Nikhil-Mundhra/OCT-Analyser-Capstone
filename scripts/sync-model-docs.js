const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.join(__dirname, '..');
const FRONTEND_PUBLIC_DOCS = path.join(ROOT_DIR, 'frontend', 'public', 'docs_content', 'models');
const FRONTEND_SRC_DOCS = path.join(ROOT_DIR, 'frontend', 'src', 'docs');

const SOURCES = [
  {
    categoryTitle: "Classification Models",
    sidebarTitle: "Classification",
    id: "classification",
    color: "purple",
    sourceDir: path.join(ROOT_DIR, 'image-classification-model-training', 'Documentation'),
    destDir: path.join(FRONTEND_PUBLIC_DOCS, 'classification')
  },
  {
    categoryTitle: "Segmentation Models",
    sidebarTitle: "Segmentation",
    id: "segmentation",
    color: "emerald",
    sourceDir: path.join(ROOT_DIR, 'image-segmentation-model-training', 'Documentation'),
    destDir: path.join(FRONTEND_PUBLIC_DOCS, 'segmentation')
  }
];

function ensureDirSync(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function extractTitle(content, filename) {
  const match = content.match(/^#\s+(.+)$/m);
  if (match) {
    return match[1].trim();
  }
  // Fallback to capitalizing the filename
  return filename
    .replace('.md', '')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function extractDescription(content) {
  // Try to find the first blockquote or the first paragraph after the title
  const lines = content.split('\n');
  let inNote = false;
  let description = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('# ')) continue;
    if (line === '') continue;

    if (line.startsWith('>')) {
      // Remove > and [!NOTE] etc
      const cleaned = line.replace(/^>\s*(\[!.*?\])?/i, '').trim();
      if (cleaned) {
        description.push(cleaned);
      }
      inNote = true;
    } else if (inNote) {
      break; // finished parsing blockquote
    } else if (line.match(/^##\s/)) {
      continue; // Skip subheadings
    } else if (description.length === 0) {
      // first normal paragraph
      description.push(line);
      break;
    }
  }
  const text = description.join(' ').substring(0, 150);
  return text.length === 150 ? text + '...' : text;
}

function processSources() {
  const generatedCategories = [];
  const slugToPathMap = {};

  SOURCES.forEach(source => {
    if (!fs.existsSync(source.sourceDir)) {
      console.warn(`Warning: Source directory not found: ${source.sourceDir}`);
      return;
    }

    ensureDirSync(source.destDir);
    const files = fs.readdirSync(source.sourceDir).filter(f => f.endsWith('.md'));
    
    const articles = [];

    files.forEach(file => {
      const srcFile = path.join(source.sourceDir, file);
      const destFile = path.join(source.destDir, file);
      
      let content = fs.readFileSync(srcFile, 'utf-8');
      
      // Rewrite internal markdown links to match generated slugs
      content = content.replace(/\]\((?!http|\/)([^)]+)\.md(#.*)?\)/g, (match, p1, p2) => {
        let cleanPath = p1.replace(/^\.\//, ''); // remove leading ./
        // Optional: also remove any directory path if they link to other folders?
        // Assuming they link to files in the same folder for now.
        return `](/docs/${source.id}-${cleanPath}${p2 || ''})`;
      });

      // Copy to public folder
      fs.writeFileSync(destFile, content);

      const title = extractTitle(content, file);
      const description = extractDescription(content);
      const slug = `${source.id}-${file.replace('.md', '').toLowerCase()}`;

      // Map slug to public path for content.js
      slugToPathMap[slug] = `/docs_content/models/${source.id}/${file}`;

      articles.push({
        slug,
        title,
        sidebarTitle: title,
        description: description || `Documentation for ${title}`
      });
    });

    if (articles.length > 0) {
      generatedCategories.push({
        id: source.id,
        title: source.categoryTitle,
        sidebarTitle: source.sidebarTitle,
        color: source.color,
        articles
      });
    }
  });

  // Write out the generated JSON
  const outputJsonPath = path.join(FRONTEND_SRC_DOCS, 'modelDocsGenerated.json');
  fs.writeFileSync(outputJsonPath, JSON.stringify({
    categories: generatedCategories,
    slugPaths: slugToPathMap
  }, null, 2));

  console.log(`Successfully synced model docs. Generated ${generatedCategories.length} categories.`);
}

processSources();
