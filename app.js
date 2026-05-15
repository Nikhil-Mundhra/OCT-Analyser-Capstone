import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
import offlineDiagram from "./Offline-training-and-validation-workflow.js";
import onlineDiagram from "./Online-clinical-inference-workflow..js";
import architectureDiagram from "./Deep-learning-model-architecture-flowchart.js";

const diagrams = {
  offline: {
    title: "Offline Training and Validation Workflow",
    caption: "Dataset ingestion, preprocessing, QC, model training, evaluation, and versioned storage.",
    source: offlineDiagram,
  },
  online: {
    title: "Online Clinical Inference Workflow",
    caption: "Clinician upload, service orchestration, QC, inference, explainability, reporting, and audit paths.",
    source: onlineDiagram,
  },
  architecture: {
    title: "Deep Learning Model Architecture",
    caption: "Preprocessing, shared 3D backbone, prediction heads, safety correction, output lanes, and report assembly.",
    source: architectureDiagram,
  },
};

const diagramGrid = document.querySelector("#diagramGrid");
const tabs = [...document.querySelectorAll(".tab")];
const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_LABEL_GAP = 18;
const LABEL_NUDGE_STEP = 8;
const MAX_LABEL_NUDGES = 12;
const LABEL_CONTAINER_PADDING_X = 6;
const LABEL_CONTAINER_PADDING_Y = 3;

mermaid.initialize({
  startOnLoad: false,
  securityLevel: "loose",
});

function getVisibleDiagramKeys(view) {
  return view === "both" ? Object.keys(diagrams) : [view];
}

function getBox(element, yOffset = 0) {
  const { x, y, width, height } = element.getBBox();

  return {
    x,
    y: y + yOffset,
    width,
    height,
    right: x + width,
    bottom: y + yOffset + height,
    centerY: y + yOffset + height / 2,
  };
}

function padBox(box, padding) {
  return {
    x: box.x - padding,
    y: box.y - padding,
    right: box.right + padding,
    bottom: box.bottom + padding,
    centerY: box.centerY,
  };
}

function boxesOverlap(a, b) {
  return a.x < b.right && a.right > b.x && a.y < b.bottom && a.bottom > b.y;
}

function getNodeBoxes(svg) {
  return [...svg.querySelectorAll("text.sequenceNumber")]
    .filter((text) => /^\d+$/.test(text.textContent.trim()))
    .map((text) => padBox(getBox(text), NODE_LABEL_GAP));
}

function applyContainerNudge(container, dy) {
  if (dy === 0) {
    return;
  }

  container.dataset.layoutNudge = String(dy);
  container.style.transformBox = "fill-box";
  container.style.transformOrigin = "center";
  container.style.transform = `translateY(${dy}px)`;
}

function createLabelBounds(label) {
  const box = getBox(label);
  const bounds = document.createElementNS(SVG_NS, "rect");

  bounds.classList.add("message-label-bounds");
  bounds.setAttribute("x", String(box.x - LABEL_CONTAINER_PADDING_X));
  bounds.setAttribute("y", String(box.y - LABEL_CONTAINER_PADDING_Y));
  bounds.setAttribute("width", String(box.width + LABEL_CONTAINER_PADDING_X * 2));
  bounds.setAttribute("height", String(box.height + LABEL_CONTAINER_PADDING_Y * 2));

  return bounds;
}

function createMessageLabelContainers(svg) {
  const labels = [...svg.querySelectorAll("text.messageText")].filter((label) => {
    return label.textContent.trim().length > 0;
  });
  const numbers = [...svg.querySelectorAll("text.sequenceNumber")].filter((text) => {
    return /^\d+$/.test(text.textContent.trim());
  });

  return labels.map((label, index) => {
    const container = document.createElementNS(SVG_NS, "g");
    const parent = label.parentNode;

    container.classList.add("message-label-container");
    container.dataset.messageIndex = String(index + 1);
    container.dataset.messageText = label.textContent.trim();
    container.dataset.sequenceNumber = numbers[index]?.textContent.trim() || "";

    parent.insertBefore(container, label);
    container.append(createLabelBounds(label), label);

    return {
      container,
      label,
      node: numbers[index] || null,
    };
  });
}

function tuneDiagramLabelSpacing(stage) {
  const svg = stage.querySelector("svg");

  if (!svg) {
    return;
  }

  const nodeBoxes = getNodeBoxes(svg);
  const containers = createMessageLabelContainers(svg);

  containers.forEach(({ container, label }) => {
    let nudge = 0;

    for (let attempt = 0; attempt < MAX_LABEL_NUDGES; attempt += 1) {
      const labelBox = getBox(label, nudge);
      const nodeBox = nodeBoxes.find((box) => boxesOverlap(labelBox, box));

      if (!nodeBox) {
        break;
      }

      const direction = labelBox.centerY <= nodeBox.centerY ? -1 : 1;
      nudge += direction * LABEL_NUDGE_STEP;
    }

    applyContainerNudge(container, nudge);
  });
}

async function renderDiagram(key) {
  const diagram = diagrams[key];
  const renderId = `diagram-${key}-${Date.now()}`;
  const article = document.createElement("article");
  article.className = "diagram-panel";
  article.innerHTML = `
    <header>
      <div>
        <h2>${diagram.title}</h2>
        <p>${diagram.caption}</p>
      </div>
    </header>
    <div class="diagram-stage" data-stage></div>
  `;

  const stage = article.querySelector("[data-stage]");

  try {
    const { svg } = await mermaid.render(renderId, diagram.source);
    stage.innerHTML = svg;
    tuneDiagramLabelSpacing(stage);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stage.innerHTML = `<pre class="render-error">${message}</pre>`;
  }

  return article;
}

async function setView(view) {
  tabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.diagram === view);
  });

  diagramGrid.classList.toggle("show-both", view === "both");
  diagramGrid.replaceChildren();

  const panels = await Promise.all(getVisibleDiagramKeys(view).map(renderDiagram));
  diagramGrid.replaceChildren(...panels);
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setView(tab.dataset.diagram);
  });
});

setView("offline");
