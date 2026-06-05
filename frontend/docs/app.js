const docs = {
  readme: {
    title: "Project README",
    url: "/public/docs/README.md",
  },
  implementation: {
    title: "Implementation Blueprint",
    url: "/public/docs/implementation-info.txt",
  },
  ipnv2: {
    title: "IPN-V2 OCTA Segmentation",
    url: "/public/docs/IPNV2_README.md",
  },
};

const params = new URLSearchParams(window.location.search);
const selected = docs[params.get("doc")] ? params.get("doc") : "readme";
const doc = docs[selected];
const title = document.getElementById("docTitle");
const content = document.getElementById("docContent");

title.textContent = doc.title;

for (const link of document.querySelectorAll(".doc-link")) {
  if (link.href === window.location.href) {
    link.classList.add("is-active");
  }
}

fetch(doc.url)
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Could not load ${doc.url}`);
    }
    return response.text();
  })
  .then((text) => {
    content.innerHTML = renderDocument(text);
  })
  .catch((error) => {
    content.innerHTML = `<p class="notice error">${escapeHtml(error.message)}</p>`;
  });

function renderDocument(text) {
  const lines = text.split(/\r?\n/);
  const html = [];
  let listOpen = false;
  let codeOpen = false;
  let codeLines = [];

  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };

  const closeCode = () => {
    if (codeOpen) {
      html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      codeLines = [];
      codeOpen = false;
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (codeOpen) {
        closeCode();
      } else {
        closeList();
        codeOpen = true;
        codeLines = [];
      }
      continue;
    }

    if (codeOpen) {
      codeLines.push(line);
      continue;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inlineMarkup(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineMarkup(bullet[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${inlineMarkup(trimmed)}</p>`);
  }

  closeCode();
  closeList();
  return html.join("");
}

function inlineMarkup(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
