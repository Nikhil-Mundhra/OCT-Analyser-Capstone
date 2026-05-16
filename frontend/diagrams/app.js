import architecture from "../Deep-learning-model-architecture-flowchart.js";
import offline from "../Offline-training-and-validation-workflow.js";
import online from "../Online-clinical-inference-workflow..js";

const diagrams = {
  offline: {
    title: "Offline Training and Validation",
    chart: offline,
  },
  online: {
    title: "Online Clinical Inference",
    chart: online,
  },
  architecture: {
    title: "Deep Learning Architecture",
    chart: architecture,
  },
};

const grid = document.getElementById("diagramGrid");
const buttons = document.querySelectorAll("[data-diagram]");

async function render(selected = "offline") {
  const mermaid = await import("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs");
  mermaid.default.initialize({ startOnLoad: false });

  const entries = selected === "both" ? Object.entries(diagrams) : [[selected, diagrams[selected]]];
  grid.innerHTML = "";

  for (const [id, diagram] of entries) {
    const section = document.createElement("section");
    section.className = "diagram-panel";
    section.innerHTML = `<h2>${diagram.title}</h2><div class="mermaid" id="diagram-${id}"></div>`;
    grid.appendChild(section);

    const { svg } = await mermaid.default.render(`rendered-${id}`, diagram.chart);
    section.querySelector(".mermaid").innerHTML = svg;
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", () => {
    buttons.forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    render(button.dataset.diagram);
  });
});

render();
