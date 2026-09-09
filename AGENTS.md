# Workspace Guidelines & Academic Reporting Invariants

## 1. UML & Architectural Diagram Invariants (OMG UML 2.5 Standards)

When generating architectural, behavioral, or design diagrams in Markdown, HTML, or report documents:

### A. UML Object Diagrams
- **Never** use `classDiagram` with bracketed labels (`class name["..."]`) or attribute assignments (`var = val`), as modern Mermaid (v10+) treats this as invalid syntax.
- **Always** use `graph TD` with semantic styling for Object Diagrams:
  ```mermaid
  graph TD
      classDef objectBox fill:#f8fafc,stroke:#334155,stroke-width:1.5px,text-align:left,font-family:monospace,font-size:11px;
      objName["<b><u>objName : ClassName</u></b><br/>----------------------------------<br/>attr1 : val1<br/>attr2 : val2"]:::objectBox
  ```
- Ensure every instantiated object has explicit association links to the rest of the object graph; do not leave floating orphan instances.

### B. UML Class Diagrams
- Every class declared in a `classDiagram` must have at least one explicit relationship (`*--`, `-->`, or `..>`). Never leave orphan classes floating unlinked.
- Avoid crossing dependency arrows over sibling classes. Group related classes with `direction TB` or sub-packages where supported.

### C. UML State Machine Diagrams (`stateDiagram-v2`)
- **Never** use `state DecisionName <<choice>>` in `stateDiagram-v2`. It triggers a known Mermaid dagre layout bug that misplaces an orphaned circle at `(0, 0)` near the start state.
- **Always** model conditional branching using direct guarded transitions between states:
  ```mermaid
  stateDiagram-v2
      StateA --> StateB: Event [GuardCondition]
      StateA --> StateC: Event [ElseCondition]
  ```

### D. Gantt Charts
- **Disable the red line:** Always include `todayMarker off` on academic, retrospective, or planned milestone Gantt charts. The default red line indicates "today's calendar date" and causes visual confusion in academic reports.
- **Prevent bottom date text collision:** For multi-month timelines rendered within standard A4 or container widths (~500–650px):
  - Do NOT use high-frequency sub-monthly ticks (like weekly `%d %b`).
  - Configure monthly intervals: `tickInterval 1month` and `axisFormat %b %Y` (or `%B`).
  - Double-check all start and end date years to prevent timeline span explosion.

### E. UML Rendering Engine Decision Boundary
- **Never** substitute generic flowchart boxes (`graph TD` with simple rectangular nodes) for UML diagrams when standard OMG UML primitives are required. Mermaid lacks native support for stick-figure actors, bullseye final nodes, 3D cuboids, ball-and-socket assembly connectors, and boundary ports.
- When generating diagrams for academic reports or formal publications requiring these primitives, **always** generate standalone vector SVG / 300 DPI PNG diagrams via dedicated Python/Edge rendering scripts (`--force-device-scale-factor=2`) to preserve vector fidelity and standard UML symbols.

### F. UML Use Case Diagrams
- **Actor Notation**: Always use standard stick figures with oval head, torso, arms, and legs (color `#d5e7cc` fill with dark border). Never use rectangular boxes or simple text labels for actors.
- **Use Cases**: Model as horizontal ellipses containing concise action-verb phrases (e.g., `Ingest Raw CSV Data`, `Train LSTM Autoencoder`).
- **System Boundary**: Enclose internal use cases within a prominent rectangular boundary box labeled with the system name (`System: <Project Name>`). Actors must reside outside the boundary.
- **Stereotyped Relationships**: Represent functional dependencies using dashed arrows with formal stereotypes: `«include»` for mandatory base sub-processes, `«extend»` for optional/conditional flows.

### G. UML Activity Diagrams
- **Initial & Final Nodes**: Always use a solid black circle (`●`) for the Initial Node and a bullseye (`◎` - solid inner circle with concentric outer ring) for the Activity Final Node.
- **Action States**: Render as rounded rectangles (`border-radius: 12px` to `16px`). Never use sharp-cornered rectangles (which represent state or class classifiers).
- **Control Nodes**: Render Decision and Merge nodes as hollow diamonds (`◇`). Every branching path must be labeled with bracketed condition guards (e.g., `[Loss <= Tolerance]`, `[Else]`).
- **Swimlanes**: Partition execution roles into vertical or horizontal swimlanes with clear header banners (e.g., `Presentation Tier`, `Pipeline Controller`, `Analytics Engine`).

### H. UML Deployment Diagrams
- **Hardware & Device Nodes**: Render execution environments and physical hardware as 3D isometric cuboid blocks with extruded top and right faces to visually convey physical/virtual computational nodes.
- **Node Stereotypes**: Label nodes with standard OMG stereotypes: `«device»` for physical machines/servers, `«executionEnvironment»` for runtime environments (e.g., Python 3.11 Runtime, CUDA 12.4 Engine).
- **Deployed Artifacts**: Model deployable files as folded-corner document rectangles with the `«artifact»` stereotype (e.g., `«artifact» app.py`, `«artifact» best_model.pt`).
- **Communication Paths**: Draw solid lines connecting node cuboids, labeled with network protocols and IPC channels (e.g., `HTTP / WebSocket (Port 8501)`, `PCIe x16 Bus (CUDA IPC)`).

### I. UML Component Diagrams
- **Component Classifier**: Enclose components in rectangles featuring the official UML 2.5 component icon in the top-right corner (rectangle with two protruding tabs) along with the `«component»` stereotype.
- **Provided Interfaces (`○-` / Lollipop)**: Draw exposed interface contracts with a solid line terminating in a circle/ball.
- **Required Interfaces (`)-` / Socket)**: Draw consumed interface dependencies with a solid line terminating in a semi-circle/socket.
- **Assembly Connectors (`○-)`)**: Model inter-component dependencies using ball-and-socket assembly connectors where the provided lollipop fits directly into the required socket, explicitly enforcing loose coupling.
- **Boundary Interaction Ports**: Mount small square ports (`▫`) directly on the boundary edges of component classifiers to anchor incoming and outgoing interface lines.
- **Physical Artifacts & Datastores**: Persisted files must be modeled as `«artifact»` (folded-corner icon) or `«datastore»` (cylinder icon).
- **Package Hierarchy**: Group components into tabbed package folders (e.g., `package Presentation Subsystem`, `package Core Analytics Subsystem`, `package Persistence Subsystem`).
- **Planar Orthogonal Routing**: All wiring must follow Manhattan orthogonal paths with dedicated channel corridors to ensure zero intersecting or crossing lines.

---

## 2. Print-Ready HTML & Table Layout Invariants

When generating standalone HTML reports or print-ready documentation:

### A. Print Margin Double-Padding Reset
- Inside `@media print`, **always** reset wrapper padding, margins, and width constraints:
  ```css
  @media print {
      .report-wrapper {
          padding: 0 !important;
          margin: 0 !important;
          max-width: 100% !important;
          width: 100% !important;
          box-shadow: none !important;
      }
      .table-container, table.report-table {
          max-width: 100% !important;
          width: 100% !important;
          overflow-x: visible !important;
      }
  }
  ```
  Leaving `.report-wrapper` screen padding active during print compounds with `@page` margins, creating double-margins that truncate wide content.

### B. Granular Table Density Binning
- Apply automatic font and padding binning based on column count:
  - $\ge 8$ columns: `.table-ultra-dense` (`font-size: 7.0pt; line-height: 1.2; padding: 2.5px 2px;`)
  - $6-7$ columns: `.table-dense` (`font-size: 8.0pt; line-height: 1.25; padding: 3px 3.5px;`)
  - $5$ columns: `.table-compact` (`font-size: 8.5pt; line-height: 1.3; padding: 4px 4px;`)
  - $1-4$ columns: Default (`font-size: 9.0pt; line-height: 1.35; padding: 5px 6px;`)
- Always include `word-break: break-word; overflow-wrap: break-word;` on table cells to ensure text wraps rather than forcing horizontal scrolling.

---

## 3. Academic Table of Contents (TOC) & Front-Matter Catalog Invariants

When rendering Table of Contents, List of Figures, List of Tables, and List of Abbreviations in academic reports and print HTML:

### A. Override Text Justification
- **Never** allow global `text-align: justify;` to apply to TOC rows. In justified text, any line wrapping will cause browser layout engines to stretch spaces between words across the full width, creating unsightly word spacing.
- **Always** force `text-align: left !important;` on `.toc-container`, `.toc-row`, and `.toc-title`.

### B. Dynamic Leader & Right-Aligned Tabular Page Numbers
- **Never** use hardcoded ASCII dot strings (`....... 25`) inside Markdown or HTML lists; they will wrap onto separate lines or break responsively on varying screen widths and print containers.
- **Always** structure TOC entries with flexbox baseline/bottom alignment:
  ```css
  .toc-row {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      text-align: left !important;
      font-size: 11pt;
      line-height: 1.35;
  }
  .toc-title { flex: 0 1 auto; max-width: 82%; text-align: left !important; }
  .toc-dots { flex: 1 1 auto; border-bottom: 1.5px dotted #666; margin: 0 6px 4px 6px; min-width: 15px; }
  .toc-page { flex: 0 0 auto; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  ```
- In `@media print`, always set `.toc-row { page-break-inside: avoid; }` to prevent individual TOC rows from fracturing across print page breaks.

### C. Standardized Hierarchy & Elimination of Bullets
- Eliminate all bullet markers (`<ul>`, `<li>`) from the Table of Contents.
- Apply consistent hierarchical indentation:
  - Level 0 (Preliminaries, Chapters, References, Bibliography, Appendices): 0 indent, bold, 12–13pt top margin on chapters.
  - Level 1 (Sections e.g. 1.1, 2.1, Appendix A): 18pt indent.
  - Level 2 (Subsections e.g. 1.3.1, 2.1.1): 36pt indent.
  - Level 3 (Sub-subsections e.g. i., ii.): 54pt indent.

### D. List of Figures (LOF) & List of Tables (LOT) Invariants
- **Never** render List of Figures or List of Tables as bordered grid data tables (`<table class="report-table">`). They are navigational catalogs, not experimental datasets.
- **Always** format LOF and LOT using the exact same flexbox structure as the Table of Contents (`.toc-container`, `.toc-row`, `.toc-dots`, `.toc-page`).
- **Always** provide active internal hyperlinks (`<a href="#fig-X-Y">` and `<a href="#tbl-X-Y">`) on both the entry title and the right-aligned page number.

### E. List of Abbreviations (LOA) Invariants
- **Never** enclose abbreviations in bordered spreadsheet grids with shaded headers.
- **Always** format abbreviations as a borderless, aligned two-column glossary (`<table class="abbreviations-table">` with `border: none !important;`).
- Ensure abbreviation terms are bold and left-aligned with fixed column width, while definition cells enforce `word-break: break-word; overflow-wrap: break-word;`.

---

## 4. Figure & Table Anchor Navigation Invariants

When generating internal cross-reference anchors for figures and diagrams:

### A. Avoid Off-Screen Figure Jump
- **Never** place the anchor ID (`id="fig-X-Y"`) solely on the bottom caption `<div class="figure-caption">`. Clicking such an anchor causes browsers to align the bottom caption with the top of the viewport, pushing the actual image or Mermaid visualization completely off-screen.
- **Always** wrap the visualization container (`.mermaid`, `.figure-container`, or `<pre>`) and its caption together inside `<div class="figure-block" id="fig-X-Y">`.
- **Always** declare `scroll-margin-top` (e.g., `70px`) on `.figure-block` so anchor navigation accounts for fixed/sticky headers or toolbars.

### B. Print Page-Break Protection for Figures
- Inside `@media print`, always declare:
  ```css
  .figure-block {
      page-break-inside: avoid !important;
      break-inside: avoid !important;
  }
  p:has(+ .figure-block) {
      page-break-after: avoid !important;
  }
  ```
  This ensures figure diagrams and their respective captions never split across separate printed pages.

---

## 5. Direct PDF Generation & Academic Pagination Invariants

When converting academic reports or print-ready HTML documents to PDF:

### A. Chromium CSS Paged Media Limitations & Deterministic Post-Processing
- **Never** rely solely on CSS `@page { @bottom-center { content: counter(page); } }` for multi-section academic documents. Chromium-based print engines (Edge, Chrome) do not support per-section dynamic counter resetting or Roman/Arabic style switching across `@page` selector scopes.
- **Always** post-process headless browser PDF exports via Python (`pypdf`):
  - **Sheet 1 (Cover Page):** Strictly unnumbered.
  - **Sheets 2 to 12 (Preliminaries):** Centered lowercase Roman numerals (`ii`, `iii`, `iv`, ..., `xii`) in Type-1 `/Times-Roman` font streams positioned at $y = 42\text{ pt}$.
  - **Sheets 13+ (Body & Appendices):** Centered Arabic numerals (`1`, `2`, `3`, ..., `N`) starting precisely at Chapter 1.
  - **Native PDF Viewer Catalog Labels (`/PageLabels`):** Inject the `/PageLabels` tree into the PDF root catalog (`/Nums [0 << /S /r /St 1 >> 12 << /S /D /St 1 >>]`) so PDF viewers (Acrobat, Edge, Preview) natively display synchronized Roman and Arabic labels in thumbnail and sidebar navigation.

### B. Consecutive Page-Break Deduplication
- **Never** allow consecutive page-break triggers (`\newpage`, `---`, `<div class="page-break"></div>`) to stack immediately adjacent to each other or precede `h1.chapter-title`.
- **Always** implement a deduplication/cleanup pass in HTML generation that collapses consecutive page breaks and removes any page break immediately preceding elements with `page-break-before: always;`. This prevents multi-page empty sheet voids.

### C. Headless Browser Print Invocation Flags
- **Always** invoke headless browser printing with:
  - `--no-pdf-header-footer`: Completely eliminates default browser date, title, and URL headers/footers.
  - `--run-all-compositor-stages-before-draw` and `--virtual-time-budget=10000`: Ensures asynchronous MathJax equations and Mermaid diagrams finish full layout and rendering before PDF vectorization.

### D. Non-Breaking Flow Containers
- **Always** apply `page-break-inside: avoid !important; break-inside: avoid !important;` to:
  - Ordered and unordered list items containing display equations (`$$...$$`).
  - Tables, algorithm containers, and code blocks (`pre`).
  - Supervisor signature blocks and consultation logs (`.supervisor-signature`).

---

## 6. Academic Cover / Title Page Distribution Invariants

When generating title or cover pages for academic reports and formal university projects:

### A. Full-Height Vertical Balance
- **Never** allow cover page content to collapse into the upper half of the page leaving a large blank void at the bottom.
- **Always** enclose the title page within a dedicated `.cover-page` container constrained to printable A4 height:
  ```css
  .cover-page {
      height: 244mm !important;
      min-height: 244mm !important;
      max-height: 245mm !important;
      display: flex !important;
      flex-direction: column !important;
      justify-content: space-between !important;
      box-sizing: border-box !important;
      page-break-after: always !important;
      break-after: page !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      text-align: center !important;
  }
  ```

### B. Three-Tier Architectural Distribution
- **Top Tier (`.cover-header`)**: Institution and Faculty names ("TRIBHUVAN UNIVERSITY", "INSTITUTE OF SCIENCE AND TECHNOLOGY") anchored to the top margin.
- **Middle Tier (`.cover-body`)**: Enforce `display: flex !important; flex-direction: column !important; justify-content: space-evenly !important; flex: 1 !important; margin: 18pt 0 15pt 0 !important;` so Title, "A PROJECT REPORT", "Submitted to:", and Degree fulfillment details are harmoniously spaced without artificial clumping.
- **Bottom Tier (`.cover-footer`)**: Two-column author & supervisor metadata grid (`.cover-meta-grid`) with `margin-bottom: 24pt`, followed by the centered submission date ("September 8, 2026") anchored to the bottom margin.

### C. Zero-Overflow Guarantee
- Every child of `.cover-page` must have `page-break-inside: avoid; break-inside: avoid; page-break-after: avoid; break-after: avoid;`.
- The total height ($244\text{ mm}$) must strictly remain below the printable page height ($246.2\text{ mm}$ for A4 with 1-inch margins) so that Page 1 never spills over into an accidental empty or fractured sheet.



