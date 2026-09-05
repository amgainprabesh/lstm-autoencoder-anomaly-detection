# Workspace Guidelines & Academic Reporting Invariants

## 1. Mermaid Diagram Invariants

When generating diagrams in Markdown, HTML, or report documents:

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

## 3. Academic Table of Contents (TOC) Invariants

When rendering Table of Contents in academic reports and print HTML:

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

