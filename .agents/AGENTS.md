# Global Agent Directives

**START HERE FOR EVERY REQUEST:**
Before taking ANY action or answering ANY question, you MUST first explicitly evaluate if Graphify or Docmancer are relevant to the request:
1. **Internal Map (Graphify):** Does this require understanding the codebase architecture, file locations, or what the next implementation steps are? If yes, check the `graphify-out/` graph or use Graphify (`/graphify query`, `/graphify path`) to orient yourself.
2. **External Guide (Docmancer):** Does this involve external libraries, APIs, or specific framework rules? If yes, use Docmancer (`docmancer query`) to retrieve the correct syntax and rules.

Only after you have established your internal and external bearings should you proceed with execution.

---

## Subagent Delegation (Parallel vs Sequential)

When assigned multiple features or bugs in a single request, you MUST decide between parallel delegation and sequential execution based on codebase overlap:

1. **Parallel Delegation (`self` subagents):** Highly beneficial for naturally isolated tasks (e.g., one agent works on frontend UI, one works on the backend database, one writes documentation). Spawn multiple `self` subagents (using `invoke_subagent`) to tackle these simultaneously.
2. **Sequential Execution (Single agent):** Detrimental for heavily intertwined tasks modifying the same files. For overlapping code changes, it is faster and safer to just knock them out sequentially yourself, ensuring you maintain the full context of the ongoing changes and avoid merge conflicts.

## Communication Directives

**No Emojis:** Do NOT use emojis anywhere—not in natural language responses, code comments, commit messages, documentation files, or markdown artifacts. Keep all communications strictly professional, concise, and clean.

## Resource Management & Execution Directives

**GPU utilization:** All scripts including training, evaluation, inference, analysis, and local app testing should utilize the GPU (`cuda` or `mps`) if available, to maximize performance.

**Script File Execution (Do Not Write Inline Python in Terminal):** NEVER execute Python logic inline via terminal commands (e.g., `python3 -c "..."` or bash heredocs). Inline terminal Python commands cause string escaping errors, IO buffering deadlocks, and truncated stack traces. ALWAYS write a dedicated Python script file (or scratch script) using `write_to_file` and execute the script file directly (e.g., `python3 test_script.py`).


## Architectural Goals

**"Best of Both Worlds" Unified Network:**
The model must function as a unified architecture with a shared encoder that simultaneously supports expert-level tissue segmentation and detailed hierarchical disease classification, specifically addressing disjoint dataset constraints (segmentation masks only available for OCT5K healthy tissues; 15-class disease labels only available in Classified without masks). The architecture must achieve this by:

1. **Shared Encoder:** Extracting universal features from both datasets.
2. **Multi-Scale Encoder Aggregation:** Pooling features from multiple encoder depths (e.g., x3, x4, x5) for the classification head to retain fine-grained spatial details lost at the bottleneck. The classification head must *not* rely on the untrained segmentation decoder for disease localization.
3. **Strict Hierarchical Classification Conditioning:** The classification branch must explicitly enforce hierarchy (e.g., L2 conditioned on L1 probabilities, L3 conditioned on L2 probabilities) via cascaded features to prevent contradictory multi-head predictions.
4. **Decoupled Decoder:** The segmentation decoder operates independently to predict tissue boundaries, without its outputs feeding back into the classification head, avoiding catastrophic failure on unseen diseases.
5. **Medical Data Augmentation Constraints:** NEVER use `RandomResizedCrop` or destructive spatial augmentations that chop off the edges of medical scans, as this acts as spatial dropout and can inadvertently erase edge-located biological markers (e.g., peripheral cysts). To eliminate UI artifacts (like compasses or logos) without risking tissue loss, strictly mandate **Segmentation-Driven Cropping**—using the U-Net to dynamically identify and preserve 100% of the retinal tissue while zeroing out the background.
