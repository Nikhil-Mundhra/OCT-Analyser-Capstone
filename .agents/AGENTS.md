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

## Design Principles

These principles guide how code should be structured in this repository. Apply judgment: the goal is maintainable, testable code, not mechanical rule following.

### Core principles (SOLID)

- **Single Responsibility**: Each module, class, or function should have one reason to change. If you find yourself describing a function with "and," it likely does too much.
- **Open/Closed**: Code should be open for extension but closed for modification. Prefer adding new code (new classes, new strategy implementations) over editing tested, working code to bolt on a special case.
- **Liskov Substitution**: A subtype must be usable anywhere its parent type is expected, without surprising behavior. If a subclass throws on a method the parent supports, the hierarchy is wrong.
- **Interface Segregation**: Prefer several small, focused interfaces over one large general purpose interface. Callers shouldn't depend on methods they never use.
- **Dependency Inversion**: High level modules should depend on abstractions, not on low level implementation details. Inject dependencies (via constructor, function args, or config) rather than reaching for globals or hardcoded instances.

### General principles

- **DRY (Don't Repeat Yourself)**: Extract shared logic once it appears three times, not on the first duplication. Premature abstraction is its own cost.
- **KISS (Keep It Simple)**: Choose the simplest design that solves the actual problem. Avoid clever solutions when a plain one reads clearly.
- **Separation of Concerns**: Keep business logic, data access, and presentation in distinct layers. A UI component shouldn't contain SQL; a data model shouldn't contain formatting logic.
- **Composition over Inheritance**: Favor combining small, focused objects over deep inheritance chains. Inheritance should model a true "is a" relationship, not just a way to reuse code.

### Structural guidelines

- **Dependency direction**: Dependencies should point inward, toward core domain logic, and outward toward infrastructure (databases, APIs, frameworks). Core logic should not import framework or infrastructure code directly.
- **File and function size**: If a file exceeds roughly 300 to 400 lines or a function exceeds 40 to 50 lines, treat that as a signal to consider splitting it, not a hard rule.
- **Naming**: Names should describe intent and behavior, not implementation. `calculateTotalPrice()` over `doStuff()`. Boolean names should read as questions: `isValid`, `hasPermission`.
- **Error handling**: Fail fast and explicitly. Don't silently swallow exceptions. Validate inputs at boundaries (API handlers, public functions) rather than scattering checks throughout internal logic.
- **Immutability by default**: Prefer immutable data structures and pure functions where practical. Mutate state only when there's a clear reason (performance, or the domain genuinely models mutable state).

### Practical rules for agents working in this codebase

- Before adding a new abstraction (interface, base class, factory), check whether an existing one already covers the need.
- When modifying existing code, match the surrounding style and structure rather than introducing a new pattern in isolation.
- Prefer small, reviewable changes over large refactors bundled with feature work.
- New modules should have a single, clearly stated purpose documented at the top of the file or in its docstring.
- Write code so that a test can be added without needing to restructure it (this usually means: inject dependencies, avoid hidden global state, keep functions pure where possible).
- All features should be built incrementally by default. Ship the smallest working version first, then extend it in follow up steps, rather than building the full scope in one pass. Only skip this and build a feature in full if explicitly told to do so.