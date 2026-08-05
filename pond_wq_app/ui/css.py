CSS = """
/* Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Title Styling */
#title h1 {
    font-size: 32px !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.small-note {
    opacity: 0.7;
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
}

/* Modern Box/Card Styling with Solid/Semi-solid colors to prevent stacking context bugs */
.box {
    position: relative;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    transition: box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                border-color 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: visible !important;
}

.dark .box {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.box:hover {
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15) !important;
    border-color: rgba(102, 126, 234, 0.4) !important;
}

.dark .box:hover {
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3) !important;
    border-color: rgba(102, 126, 234, 0.6) !important;
}

/* Enhanced Dataset Cards */
.dataset-container {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: flex-start;
    margin-top: 12px;
}

.dataset-card {
    flex: 0 1 220px;
    border-radius: 16px;
    padding: 18px;
    color: white;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.dataset-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.dataset-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}

.dataset-card:hover::before {
    opacity: 1;
}

.dataset-card h3 {
    margin: 0 0 12px 0;
    font-size: 17px;
    font-weight: 600;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.dataset-card p {
    margin: 6px 0;
    font-size: 13px;
    text-align: center;
    font-weight: 500;
}

/* Button Enhancements */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Section Headers */
h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 18px;
    color: #1e293b;
    margin-bottom: 12px;
}

.dark h3 {
    color: #f1f5f9;
}

/* Tabs Styling */
.tab-nav button {
    border-radius: 8px 8px 0 0;
    transition: all 0.2s ease;
}

.tab-nav button:hover {
    background: rgba(102, 126, 234, 0.1);
}

/* Input Controls */
input[type="range"] {
    accent-color: #667eea;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Model Status Cards */
.model-status-card {
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(249,250,251,0.9) 100%);
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.dark .model-status-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.9) 100%);
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}

.model-status-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}

.best-model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-left: 8px;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.download-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #667eea;
    text-decoration: none;
    font-weight: 600;
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(102, 126, 234, 0.1);
    transition: all 0.2s ease;
    margin-top: 8px;
}

.download-link:hover {
    background: rgba(102, 126, 234, 0.2);
    transform: translateX(2px);
}

/* Gradient Accents */
.gradient-accent {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    height: 3px;
    border-radius: 2px;
    margin: 16px 0;
}

/* Animation Keyframes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.5s ease-out;
}

/* Only hide plots that are marked as hidden */
.gr-plot.hidden,
div[data-testid="plot"].hidden {
    display: none !important;
}

/* =========================================================
   DROPDOWN FIX — Gradio 4.x compatible
   The dropdown list is rendered as a <ul> with role="listbox"
   inside a fixed/absolute wrapper. We ensure it always floats
   above all content by using a very high z-index and making
   sure NO ancestor has overflow:hidden or transform.
   ========================================================= */

/* Strip overflow-hidden from all Gradio wrapper containers */
.gradio-column,
.gr-form,
.gr-box,
.form,
.gap,
.flex,
.flex-col {
    overflow: visible !important;
}

/* Gradio 4.x dropdown wrapper — contains the open list */
[data-testid="dropdown"],
.gr-dropdown {
    position: relative !important;
    z-index: 9000 !important;
    overflow: visible !important;
}

/* The open dropdown list (Gradio 4 renders a <ul role=listbox>)
   position:fixed + JS-computed coords so it always escapes overflow:hidden ancestors */
[data-testid="dropdown"] ul,
[data-testid="dropdown"] [role="listbox"],
[data-testid="dropdown"] .options,
.gr-dropdown ul,
.gr-dropdown [role="listbox"],
.gr-dropdown .options,
ul[role="listbox"] {
    position: fixed !important;
    z-index: 999999 !important;
    height: auto !important;
    max-height: 280px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: block !important;
    background: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.22), 0 4px 16px rgba(0,0,0,0.12) !important;
    padding: 8px !important;
}

/* Free the immediate parent wrappers of the listbox from Svelte-injected height/overflow constraints */
ul[role="listbox"] *,
[data-testid="dropdown"] ul *,
[data-testid="dropdown"] [role="listbox"] * {
    overflow: visible !important;
}

/* The Svelte wrapper div that contains the ul — must not clip it */
[data-testid="dropdown"] > div,
[data-testid="dropdown"] > div > div,
[data-testid="dropdown"] .wrap > div {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
}

/* Dark mode support for dropdown list */
.dark [data-testid="dropdown"] ul,
.dark [data-testid="dropdown"] [role="listbox"],
.dark [data-testid="dropdown"] .options,
.dark .gr-dropdown ul,
.dark .gr-dropdown [role="listbox"],
.dark .gr-dropdown .options,
.dark ul[role="listbox"] {
    background: #1e293b !important;
    border: 1.5px solid #4338ca !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}

/* Individual dropdown list items */
[data-testid="dropdown"] ul li,
[data-testid="dropdown"] [role="option"],
.gr-dropdown ul li,
ul[role="listbox"] li {
    padding: 9px 14px !important;
    border-radius: 7px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    color: #1e293b !important;
    transition: background 0.15s ease !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: ellipsis !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Dark mode for list items */
.dark [data-testid="dropdown"] ul li,
.dark [data-testid="dropdown"] [role="option"],
.dark .gr-dropdown ul li,
.dark ul[role="listbox"] li {
    color: #f1f5f9 !important;
}

/* Hover state for list items */
[data-testid="dropdown"] ul li:hover,
[data-testid="dropdown"] [role="option"]:hover,
.gr-dropdown ul li:hover,
ul[role="listbox"] li:hover {
    background: rgba(102, 126, 234, 0.1) !important;
    color: #4338ca !important;
}

.dark [data-testid="dropdown"] ul li:hover,
.dark [data-testid="dropdown"] [role="option"]:hover,
.dark .gr-dropdown ul li:hover,
.dark ul[role="listbox"] li:hover {
    background: rgba(102, 126, 234, 0.25) !important;
    color: #c7d2fe !important;
}

/* Selected/checked item */
[data-testid="dropdown"] ul li[aria-selected="true"],
[data-testid="dropdown"] [role="option"][aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%) !important;
    color: #4338ca !important;
    font-weight: 600 !important;
}

.dark [data-testid="dropdown"] ul li[aria-selected="true"],
.dark [data-testid="dropdown"] [role="option"][aria-selected="true"],
.dark ul[role="listbox"] li[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.3) 0%, rgba(118,75,162,0.3) 100%) !important;
    color: #c7d2fe !important;
}

/* Multiselect selected tags/pills in the input area */
[data-testid="dropdown"] .token,
.gr-dropdown .token,
[data-testid="dropdown"] [data-tag],
.multiselect .tag,
.svelte-multiselect .token {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 3px 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin: 2px !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3) !important;
}

/* Scrollbar inside dropdown list */
ul[role="listbox"]::-webkit-scrollbar,
[data-testid="dropdown"] ul::-webkit-scrollbar {
    width: 5px;
}

ul[role="listbox"]::-webkit-scrollbar-thumb,
[data-testid="dropdown"] ul::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 3px;
}

/* =============================================================
   CRITICAL GRADIO 4.x / SVELTE OVERFLOW FIX
   Gradio 4.x wraps every component in Svelte-generated divs
   that may have overflow:hidden, clipping dropdown popups.
   We must force overflow:visible on ALL possible ancestors.
   ============================================================= */

/* Target all Svelte wrapper divs by partial class name match */
[class*="svelte-"] {
    overflow: visible !important;
}

/* Gradio block-level containers */
.block,
.gr-block,
.gr-group,
.gr-form,
.gr-padded,
.gr-compact,
.gr-panel,
.gr-box {
    overflow: visible !important;
}

/* Row and column wrappers */
.row,
.gr-row,
.col,
.gr-col,
.gradio-row,
.gradio-column {
    overflow: visible !important;
}

/* Accordion content — must not clip dropdowns inside */
.accordion,
.gr-accordion,
details > div,
.details-body {
    overflow: visible !important;
}

/* The multiselect wrapper itself must allow overflow */
.multiselect,
.wrap,
.wrap-inner,
.token-container,
.selected-items {
    overflow: visible !important;
}

/* Ensure items in the listbox do NOT have overflow:hidden that clips text */
ul[role="listbox"] li,
[role="option"] {
    overflow: visible !important;
    white-space: normal !important;
}

/* Gradio Tabs and Accordions overflow visible */
.tabs,
.tabitem,
.tab-content,
.tabitem > div,
.gradio-container {
    overflow: visible !important;
}

/* =============================================================
   PROFESSIONAL THEME POLISH — LIGHT + DARK
   ============================================================= */
.gradio-container {
    max-width: 1580px !important;
    margin: 0 auto !important;
    padding: 18px 22px 32px !important;
    background: var(--body-background-fill) !important;
    color: var(--body-text-color) !important;
}

#title {
    position: relative;
    overflow: hidden;
    padding: 24px 28px;
    margin: 4px 0 20px;
    border: 1px solid var(--block-border-color);
    border-radius: 20px;
    background:
        radial-gradient(circle at top right, rgba(99,102,241,.18), transparent 38%),
        linear-gradient(135deg, var(--block-background-fill), var(--background-fill-secondary));
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
}

#title h1 {
    margin: 0 0 7px !important;
    line-height: 1.18 !important;
}

#title p {
    margin: 0;
    color: var(--body-text-color-subdued) !important;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: .2px;
}

.box {
    background: var(--block-background-fill) !important;
    border-color: var(--block-border-color) !important;
    color: var(--body-text-color) !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
}

.box:hover {
    box-shadow: 0 12px 30px rgba(79, 70, 229, 0.12) !important;
}

.box h1,
.box h2,
.box h3,
.box h4,
.box p,
.box span,
.box label,
.box strong,
.box table,
.box th,
.box td {
    color: inherit;
}

.compact-action button,
button.compact-action {
    border-radius: 10px !important;
    min-height: 38px !important;
    font-weight: 650 !important;
}

.refresh-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 9px;
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--primary-500) 28%, var(--block-border-color));
    background: color-mix(in srgb, var(--primary-500) 9%, var(--block-background-fill));
    color: var(--body-text-color);
    font-size: 12px;
    font-weight: 650;
}

.refresh-chip-warning {
    border-color: color-mix(in srgb, #f59e0b 40%, var(--block-border-color));
    background: color-mix(in srgb, #f59e0b 10%, var(--block-background-fill));
}

.empty-state {
    padding: 22px;
    text-align: center;
    color: var(--body-text-color-subdued);
    border: 1px dashed var(--block-border-color);
    border-radius: 14px;
    background: var(--background-fill-secondary);
}

.muted-copy {
    color: var(--body-text-color-subdued) !important;
}

.status-card {
    border-radius: 16px;
    padding: 18px;
    border: 1px solid var(--block-border-color);
    background: var(--block-background-fill);
    color: var(--body-text-color);
}

.status-card-success {
    border-left: 4px solid #10b981;
    background:
        linear-gradient(135deg,
            color-mix(in srgb, #10b981 8%, var(--block-background-fill)),
            var(--block-background-fill));
}

.status-card-title {
    margin-bottom: 14px;
    font-size: 16px;
    font-weight: 750;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.status-grid > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 10px 12px;
    border-radius: 11px;
    background: var(--background-fill-secondary);
}

.status-grid span {
    color: var(--body-text-color-subdued);
    font-size: 11px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .55px;
}

.status-card-detail {
    margin-top: 12px;
    line-height: 1.7;
    font-size: 13px;
}

.status-card-foot {
    margin-top: 12px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}

.download-panel {
    padding: 18px 20px;
    border: 1px solid var(--block-border-color);
    border-radius: 16px;
    background:
        linear-gradient(135deg,
            color-mix(in srgb, var(--primary-500) 8%, var(--block-background-fill)),
            var(--block-background-fill));
    color: var(--body-text-color);
}

.download-panel-title {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 6px;
    font-size: 16px;
    font-weight: 750;
}

.download-panel-copy {
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 13px;
    line-height: 1.6;
}

.gradio-container button {
    border-radius: 11px !important;
    font-weight: 650 !important;
}

.gradio-container .primary {
    box-shadow: 0 6px 18px rgba(79, 70, 229, .2);
}

/* Keep legacy inline light-card HTML readable in dark theme. */
.dark [style*="color:#1e293b"],
.dark [style*="color: #1e293b"],
.dark [style*="color:#334155"],
.dark [style*="color: #334155"],
.dark [style*="color:#475569"],
.dark [style*="color: #475569"] {
    color: #e2e8f0 !important;
}

.dark [style*="color:#64748b"],
.dark [style*="color: #64748b"],
.dark [style*="color:gray"],
.dark [style*="color: gray"] {
    color: #a8b3c7 !important;
}

.dark [style*="background:#fff"],
.dark [style*="background: #fff"],
.dark [style*="background:#ffffff"],
.dark [style*="background: #ffffff"] {
    background: #1e293b !important;
}

.dark [style*="background:#f8fafc"],
.dark [style*="background: #f8fafc"],
.dark [style*="background:#f0f7ff"],
.dark [style*="background-color: #f0f7ff"] {
    background: #172033 !important;
}

.dark table,
.dark th,
.dark td {
    color: #e2e8f0;
    border-color: #334155 !important;
}

@media (max-width: 760px) {
    .gradio-container {
        padding: 10px 10px 24px !important;
    }

    #title {
        padding: 20px;
        border-radius: 16px;
    }

    #title h1 {
        font-size: 25px !important;
    }

    .status-grid {
        grid-template-columns: 1fr;
    }
}

/* =============================================================
   COMPACT MODERN OPERATIONAL WORKSPACES
   ============================================================= */
:root {
    --ui-cyan: #06b6d4;
    --ui-blue: #3b82f6;
    --ui-violet: #7c3aed;
    --ui-rose: #ec4899;
    --ui-green: #10b981;
    --ui-amber: #f59e0b;
}

.gradio-container {
    max-width: 1500px !important;
    padding: 12px 18px 28px !important;
}

#title {
    padding: 20px 24px !important;
    margin-bottom: 12px !important;
    border-radius: 22px !important;
}

#title h1 {
    font-size: clamp(25px, 2vw, 34px) !important;
}

.workspace-hero {
    position: relative;
    display: flex;
    align-items: center;
    gap: 15px;
    min-height: 92px;
    padding: 17px 20px;
    margin: 7px 0 14px;
    border: 1px solid color-mix(in srgb, var(--primary-500) 20%, var(--block-border-color));
    border-radius: 24px;
    overflow: hidden;
    background:
        radial-gradient(circle at 88% -20%, rgba(124,58,237,.28), transparent 35%),
        linear-gradient(135deg,
            color-mix(in srgb, #6366f1 8%, var(--block-background-fill)),
            color-mix(in srgb, #06b6d4 5%, var(--block-background-fill)));
    box-shadow: 0 10px 28px rgba(15,23,42,.07);
}

.water-hero {
    background:
        radial-gradient(circle at 88% -20%, rgba(6,182,212,.28), transparent 35%),
        linear-gradient(135deg,
            color-mix(in srgb, #0ea5e9 8%, var(--block-background-fill)),
            color-mix(in srgb, #10b981 5%, var(--block-background-fill)));
}

.workspace-hero::after {
    content: "";
    position: absolute;
    right: -40px;
    bottom: -60px;
    width: 210px;
    height: 210px;
    border: 28px solid rgba(255,255,255,.11);
    border-radius: 50%;
    pointer-events: none;
}

.workspace-hero-icon {
    display: grid;
    place-items: center;
    flex: 0 0 52px;
    width: 52px;
    height: 52px;
    border-radius: 17px;
    color: white;
    font-size: 25px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 10px 22px rgba(99,102,241,.27);
}

.water-hero .workspace-hero-icon {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    box-shadow: 0 10px 22px rgba(6,182,212,.25);
}

.workspace-kicker,
.eyebrow {
    color: color-mix(in srgb, var(--primary-500) 85%, var(--body-text-color));
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.35px;
    text-transform: uppercase;
}

.workspace-title,
.results-title {
    margin-top: 2px;
    color: var(--body-text-color);
    font-family: 'Poppins', sans-serif;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.25;
}

.workspace-copy {
    margin-top: 3px;
    color: var(--body-text-color-subdued);
    font-size: 12.5px;
}

.workspace-badge {
    z-index: 1;
    margin-left: auto;
    padding: 7px 11px;
    border: 1px solid rgba(99,102,241,.22);
    border-radius: 999px;
    background: color-mix(in srgb, var(--primary-500) 10%, var(--block-background-fill));
    color: var(--body-text-color);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .9px;
    white-space: nowrap;
}

.operational-grid {
    align-items: flex-start !important;
    gap: 13px !important;
}

.ui-card {
    position: relative;
    gap: 10px !important;
    min-width: 0;
    padding: 15px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 21px;
    background: var(--block-background-fill);
    box-shadow: 0 8px 24px rgba(15,23,42,.055);
    overflow: visible !important;
}

.ui-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 20px;
    right: 20px;
    height: 3px;
    border-radius: 0 0 999px 999px;
    background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
}

.accent-cyan { --accent-a: #06b6d4; --accent-b: #3b82f6; }
.accent-blue { --accent-a: #3b82f6; --accent-b: #6366f1; }
.accent-violet { --accent-a: #6366f1; --accent-b: #8b5cf6; }
.accent-rose { --accent-a: #ec4899; --accent-b: #f97316; }

.card-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 2px;
}

.card-icon {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    flex: 0 0 36px;
    border-radius: 12px;
    background: color-mix(in srgb, var(--accent-a) 12%, var(--block-background-fill));
    font-size: 18px;
}

.card-heading > div {
    display: flex;
    flex-direction: column;
    line-height: 1.2;
}

.card-heading b {
    color: var(--body-text-color);
    font-size: 14px;
    font-weight: 750;
}

.card-step {
    color: var(--body-text-color-subdued);
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.05px;
}

.card-helper {
    margin: -2px 0 3px !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 11.5px !important;
    line-height: 1.45 !important;
}

.ui-card > .block,
.ui-card > .form,
.ui-card .block {
    margin: 0 !important;
}

.ui-card .form {
    gap: 8px !important;
}

.compact-tabs > .tab-nav {
    gap: 4px !important;
    padding: 3px !important;
    border-radius: 12px !important;
    background: var(--background-fill-secondary) !important;
}

.compact-tabs > .tab-nav button {
    min-height: 32px !important;
    padding: 5px 10px !important;
    border-radius: 9px !important;
    font-size: 12px !important;
}

.soft-accordion {
    margin: 0 !important;
    border-radius: 12px !important;
}

.soft-button button,
button.soft-button {
    min-height: 37px !important;
    border-radius: 11px !important;
    border-color: color-mix(in srgb, var(--primary-500) 20%, var(--block-border-color)) !important;
    background: color-mix(in srgb, var(--primary-500) 5%, var(--block-background-fill)) !important;
}

.class-selector label {
    font-size: 12px !important;
}

.class-selector .wrap {
    gap: 6px !important;
}

.class-selector label span {
    border-radius: 999px !important;
}

.model-profile,
.dataset-ready-card,
.prediction-overview {
    color: var(--body-text-color);
    border: 1px solid var(--block-border-color);
    border-radius: 16px;
    background:
        linear-gradient(145deg,
            color-mix(in srgb, var(--primary-500) 5%, var(--block-background-fill)),
            var(--block-background-fill));
}

.model-profile {
    padding: 13px;
}

.model-profile-water {
    background:
        linear-gradient(145deg,
            color-mix(in srgb, #06b6d4 6%, var(--block-background-fill)),
            var(--block-background-fill));
}

.model-profile-top,
.dataset-ready-head,
.prediction-overview-head {
    display: flex;
    align-items: center;
    gap: 10px;
}

.model-avatar {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    border-radius: 13px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 19px;
    box-shadow: 0 7px 16px rgba(99,102,241,.22);
}

.model-profile-water .model-avatar {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
}

.model-profile-heading {
    min-width: 0;
    flex: 1;
}

.model-profile-title {
    overflow: hidden;
    color: var(--body-text-color);
    font-size: 13px;
    font-weight: 750;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.model-profile-name {
    overflow: hidden;
    margin-top: 1px;
    color: var(--body-text-color-subdued);
    font-size: 10.5px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.model-state-badge {
    padding: 4px 7px;
    border-radius: 999px;
    background: color-mix(in srgb, #10b981 13%, var(--block-background-fill));
    color: #059669;
    font-size: 8.5px;
    font-weight: 850;
    letter-spacing: .8px;
}

.dark .model-state-badge { color: #6ee7b7; }

.model-facts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
    margin-top: 10px;
}

.model-facts > div {
    min-width: 0;
    padding: 8px 9px;
    border-radius: 10px;
    background: var(--background-fill-secondary);
}

.model-facts span {
    display: block;
    color: var(--body-text-color-subdued);
    font-size: 8.5px;
    font-weight: 800;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.model-facts b {
    display: block;
    overflow: hidden;
    margin-top: 2px;
    color: var(--body-text-color);
    font-size: 10.5px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.model-profile-copy {
    margin-top: 9px;
    color: var(--body-text-color-subdued);
    font-size: 11px;
    line-height: 1.45;
}

.class-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 9px;
}

.class-pill,
.metric-pill {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 4px 8px;
    border: 1px solid color-mix(in srgb, var(--primary-500) 20%, var(--block-border-color));
    border-radius: 999px;
    background: color-mix(in srgb, var(--primary-500) 7%, var(--block-background-fill));
    color: var(--body-text-color);
    font-size: 9.5px;
    font-weight: 650;
}

.metric-pill {
    border-color: color-mix(in srgb, #06b6d4 24%, var(--block-border-color));
    background: color-mix(in srgb, #06b6d4 8%, var(--block-background-fill));
}

.class-pill-muted,
.metric-pill-muted {
    color: var(--body-text-color-subdued);
}

.model-profile-foot {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-top: 9px;
    padding-top: 8px;
    border-top: 1px solid var(--block-border-color);
    color: var(--body-text-color-subdued);
    font-size: 8.5px;
}

.model-profile-foot span {
    overflow: hidden;
    max-width: 48%;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.mini-empty {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 58px;
    padding: 10px 11px;
    border: 1px dashed var(--block-border-color);
    border-radius: 13px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
}

.mini-empty-icon {
    display: grid;
    place-items: center;
    flex: 0 0 31px;
    width: 31px;
    height: 31px;
    border-radius: 10px;
    background: color-mix(in srgb, var(--primary-500) 10%, var(--block-background-fill));
}

.mini-empty b,
.mini-empty small {
    display: block;
}

.mini-empty b {
    font-size: 11.5px;
}

.mini-empty small {
    margin-top: 2px;
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
    line-height: 1.35;
}

.inline-alert {
    padding: 10px 12px;
    border-radius: 12px;
    font-size: 11px;
}

.inline-alert-error {
    border: 1px solid color-mix(in srgb, #ef4444 35%, var(--block-border-color));
    background: color-mix(in srgb, #ef4444 8%, var(--block-background-fill));
    color: color-mix(in srgb, #ef4444 78%, var(--body-text-color));
}

.dataset-ready-card {
    padding: 12px;
    background:
        linear-gradient(145deg,
            color-mix(in srgb, #10b981 7%, var(--block-background-fill)),
            var(--block-background-fill));
}

.dataset-ready-head {
    justify-content: space-between;
}

.dataset-ready-title {
    color: var(--body-text-color);
    font-size: 14px;
    font-weight: 780;
}

.success-orb {
    display: grid;
    place-items: center;
    width: 31px;
    height: 31px;
    border-radius: 11px;
    background: linear-gradient(135deg, #10b981, #14b8a6);
    color: white;
    font-weight: 850;
    box-shadow: 0 7px 16px rgba(16,185,129,.2);
}

.compact-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
    margin-top: 9px;
}

.compact-stat-grid > div {
    min-width: 0;
    padding: 7px 8px;
    border-radius: 10px;
    background: var(--background-fill-secondary);
}

.compact-stat-grid span {
    display: block;
    color: var(--body-text-color-subdued);
    font-size: 8px;
    font-weight: 800;
    letter-spacing: .55px;
    text-transform: uppercase;
}

.compact-stat-grid b {
    display: block;
    overflow: hidden;
    margin-top: 2px;
    color: var(--body-text-color);
    font-size: 10px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.dataset-ready-meta {
    margin-top: 8px;
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
}

.action-dock {
    align-items: center !important;
    gap: 12px !important;
    margin: 13px 0 16px !important;
    padding: 11px 13px !important;
    border: 1px solid color-mix(in srgb, #6366f1 22%, var(--block-border-color));
    border-radius: 18px;
    background:
        linear-gradient(135deg,
            color-mix(in srgb, #6366f1 8%, var(--block-background-fill)),
            color-mix(in srgb, #ec4899 4%, var(--block-background-fill)));
    box-shadow: 0 8px 22px rgba(99,102,241,.08);
}

.action-dock-water {
    border-color: color-mix(in srgb, #06b6d4 24%, var(--block-border-color));
    background:
        linear-gradient(135deg,
            color-mix(in srgb, #06b6d4 8%, var(--block-background-fill)),
            color-mix(in srgb, #10b981 4%, var(--block-background-fill)));
}

.action-dock > .html-container {
    flex: 1 1 auto !important;
}

.action-dock-copy {
    display: flex;
    align-items: center;
    gap: 10px;
}

.action-orb {
    display: grid;
    place-items: center;
    width: 35px;
    height: 35px;
    flex: 0 0 35px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    color: white;
}

.action-dock-water .action-orb {
    background: linear-gradient(135deg, #06b6d4, #10b981);
}

.action-dock-copy b,
.action-dock-copy small {
    display: block;
}

.action-dock-copy b {
    color: var(--body-text-color);
    font-size: 12px;
}

.action-dock-copy small {
    margin-top: 1px;
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
}

.hero-action {
    flex: 0 0 min(330px, 35%) !important;
}

.hero-action button,
button.hero-action {
    min-height: 46px !important;
    border: 0 !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-size: 13.5px !important;
    font-weight: 780 !important;
    box-shadow: 0 9px 20px rgba(99,102,241,.26) !important;
}

.action-dock-water .hero-action button {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    box-shadow: 0 9px 20px rgba(6,182,212,.24) !important;
}

.results-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 12px;
    margin: 2px 2px 9px;
}

.results-hint {
    color: var(--body-text-color-subdued);
    font-size: 10px;
}

.results-stage {
    gap: 10px !important;
    padding: 14px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 22px;
    background: var(--block-background-fill);
    box-shadow: 0 9px 26px rgba(15,23,42,.055);
}

.results-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 105px;
    padding: 18px;
    border: 1px dashed color-mix(in srgb, var(--primary-500) 28%, var(--block-border-color));
    border-radius: 17px;
    background:
        radial-gradient(circle at top center,
            color-mix(in srgb, var(--primary-500) 9%, transparent), transparent 56%),
        var(--background-fill-secondary);
    text-align: center;
}

.results-empty > span {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin-bottom: 7px;
    border-radius: 13px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 18px;
}

.results-empty b {
    color: var(--body-text-color);
    font-size: 12.5px;
}

.results-empty small {
    margin-top: 3px;
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
}

.prediction-overview {
    padding: 15px;
}

.prediction-overview-head {
    justify-content: space-between;
    margin-bottom: 11px;
}

.total-badge {
    display: flex;
    align-items: baseline;
    gap: 4px;
    padding: 7px 10px;
    border-radius: 12px;
    background: var(--background-fill-secondary);
}

.total-badge b {
    color: var(--body-text-color);
    font-size: 16px;
}

.total-badge span {
    color: var(--body-text-color-subdued);
    font-size: 9px;
}

.prediction-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 9px;
}

.prediction-class-card {
    position: relative;
    min-width: 0;
    padding: 12px;
    border: 1px solid color-mix(in srgb, var(--card-a) 24%, var(--block-border-color));
    border-radius: 15px;
    overflow: hidden;
    background:
        radial-gradient(circle at 110% -20%, color-mix(in srgb, var(--card-b) 22%, transparent), transparent 48%),
        color-mix(in srgb, var(--card-a) 5%, var(--block-background-fill));
}

.prediction-class-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, var(--card-a), var(--card-b));
}

.prediction-class-top {
    display: flex;
    justify-content: space-between;
    gap: 7px;
}

.prediction-class-name {
    overflow: hidden;
    color: var(--body-text-color);
    font-size: 10.5px;
    font-weight: 750;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.prediction-class-percent {
    color: color-mix(in srgb, var(--card-a) 80%, var(--body-text-color));
    font-size: 9px;
    font-weight: 800;
}

.prediction-class-count {
    margin-top: 7px;
    color: var(--body-text-color);
    font-size: 25px;
    font-weight: 800;
    line-height: 1;
}

.prediction-class-caption {
    margin-top: 2px;
    color: var(--body-text-color-subdued);
    font-size: 8.5px;
}

.prediction-progress {
    height: 4px;
    margin-top: 9px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--card-a) 12%, var(--background-fill-secondary));
    overflow: hidden;
}

.prediction-progress span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--card-a), var(--card-b));
}

.results-tabs > .tab-nav {
    padding: 4px !important;
    border-radius: 13px !important;
    background: var(--background-fill-secondary) !important;
}

.results-tabs > .tab-nav button {
    min-height: 34px !important;
    border-radius: 10px !important;
    font-size: 11px !important;
}

.download-panel {
    padding: 14px 16px !important;
    border-radius: 15px !important;
}

@media (max-width: 1000px) {
    .workspace-badge {
        display: none;
    }

    .operational-grid {
        flex-wrap: wrap !important;
    }

    .operational-grid > div {
        min-width: min(100%, 360px) !important;
    }

    .compact-stat-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .workspace-hero {
        align-items: flex-start;
        padding: 15px;
        border-radius: 18px;
    }

    .workspace-hero-icon {
        width: 43px;
        height: 43px;
        flex-basis: 43px;
        border-radius: 14px;
    }

    .workspace-title {
        font-size: 17px;
    }

    .workspace-copy,
    .action-dock-copy {
        display: none;
    }

    .action-dock {
        padding: 8px !important;
    }

    .hero-action {
        flex: 1 1 100% !important;
    }

    .results-heading {
        align-items: flex-start;
        flex-direction: column;
    }

    .prediction-card-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}


/* =============================================================
   MODERN ADVANCED TRAINING WORKSPACE
   ============================================================= */
.advanced-hero {
    background:
        radial-gradient(circle at 88% -20%, rgba(236,72,153,.22), transparent 35%),
        linear-gradient(135deg,
            color-mix(in srgb, #6366f1 9%, var(--block-background-fill)),
            color-mix(in srgb, #f59e0b 5%, var(--block-background-fill)));
}

.advanced-hero-icon {
    background: linear-gradient(135deg, #6366f1, #ec4899);
    box-shadow: 0 10px 22px rgba(99,102,241,.25);
}

.advanced-top-grid,
.advanced-config-grid,
.advanced-results-grid {
    align-items: flex-start !important;
    gap: 14px !important;
}

.advanced-step-card {
    min-height: 100%;
}

.advanced-card-index {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    flex: 0 0 36px;
    border: 1px solid color-mix(in srgb, var(--accent-a) 28%, var(--block-border-color));
    border-radius: 12px;
    background: color-mix(in srgb, var(--accent-a) 11%, var(--block-background-fill));
    color: color-mix(in srgb, var(--accent-a) 88%, var(--body-text-color));
    font-size: 14px;
    font-weight: 850;
}

.advanced-section-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 16px;
    margin: 18px 3px 10px;
}

.advanced-section-heading > span {
    color: var(--body-text-color-subdued);
    font-size: 12.5px;
}

.advanced-config-main,
.advanced-side-stack {
    gap: 14px !important;
}

.config-card {
    gap: 10px !important;
    padding: 17px !important;
}

.config-card-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 4px;
}

.config-card-heading > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.config-card-heading b {
    color: var(--body-text-color);
    font-size: 16px;
    font-weight: 780;
}

.feature-family-intro {
    margin: -2px 0 3px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
    line-height: 1.45;
}

.feature-toggle-row {
    align-items: stretch !important;
    gap: 10px !important;
    margin: 0 !important;
}

.feature-toggle-row + .feature-toggle-row {
    margin-top: 2px !important;
}

.feature-toggle-cell {
    min-width: 180px !important;
    padding: 9px 11px !important;
    border: 1px solid color-mix(in srgb, #06b6d4 20%, var(--block-border-color));
    border-radius: 13px;
    background: color-mix(in srgb, #06b6d4 5%, var(--background-fill-secondary));
    transition: border-color .18s ease, background .18s ease, transform .18s ease;
}

.feature-toggle-cell:hover {
    transform: translateY(-1px);
    border-color: color-mix(in srgb, #06b6d4 42%, var(--block-border-color));
    background: color-mix(in srgb, #06b6d4 8%, var(--background-fill-secondary));
}

.feature-toggle-cell > div,
.feature-toggle-cell .form,
.feature-toggle-cell label {
    min-width: 0 !important;
    width: 100% !important;
}

.feature-toggle-cell label {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    line-height: 1.35 !important;
}

.feature-toggle-cell input[type="checkbox"] {
    flex: 0 0 auto !important;
}

.feature-toggle-row-last > div {
    flex: 1 1 0 !important;
}

.sub-config-card {
    gap: 9px !important;
    margin-top: 6px !important;
    padding: 13px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 15px;
    background: var(--background-fill-secondary);
}

.sub-config-card h4 {
    margin: 0 0 3px !important;
    color: var(--body-text-color) !important;
    font-size: 14px !important;
}

.sticky-training-card,
.training-setup-card {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
}

.advanced-side-stack {
    position: relative !important;
    isolation: isolate;
}

.training-action-card {
    position: relative !important;
    z-index: 0 !important;
    margin-top: 0 !important;
}

.training-action-card {
    gap: 10px !important;
    padding: 17px !important;
    border: 1px solid color-mix(in srgb, #ec4899 24%, var(--block-border-color));
    border-radius: 21px;
    background:
        radial-gradient(circle at 110% -25%, rgba(236,72,153,.16), transparent 44%),
        var(--block-background-fill);
    box-shadow: 0 8px 24px rgba(15,23,42,.055);
}

.advanced-run-buttons {
    gap: 9px !important;
}

.advanced-run-buttons > div:first-child {
    flex: 0 0 34% !important;
}

.advanced-run-buttons > div:last-child {
    flex: 1 1 auto !important;
}

.advanced-train-button {
    flex: 1 1 auto !important;
}

.advanced-train-button button,
button.advanced-train-button {
    min-height: 48px !important;
    background: linear-gradient(135deg, #6366f1, #ec4899) !important;
    box-shadow: 0 9px 22px rgba(99,102,241,.25) !important;
}

.advanced-results-heading {
    margin-top: 21px;
}

.advanced-result-panel {
    min-width: 0;
    padding: 15px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 21px;
    background: var(--block-background-fill);
    box-shadow: 0 8px 24px rgba(15,23,42,.055);
}

.advanced-result-panel > .html-container {
    width: 100%;
}

.advanced-empty-state {
    display: flex;
    align-items: center;
    gap: 13px;
    min-height: 84px;
    padding: 16px;
    border: 1px dashed color-mix(in srgb, var(--primary-500) 28%, var(--block-border-color));
    border-radius: 16px;
    background: var(--background-fill-secondary);
}

.advanced-empty-mark {
    display: grid;
    place-items: center;
    min-width: 44px;
    height: 44px;
    padding: 0 9px;
    border-radius: 13px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: .5px;
}

.advanced-empty-state b,
.advanced-empty-state small {
    display: block;
}

.advanced-empty-state b {
    color: var(--body-text-color);
    font-size: 14.5px;
}

.advanced-empty-state small {
    margin-top: 4px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
    line-height: 1.45;
}

.training-history-panel {
    width: 100%;
}

.training-history-head,
.training-summary-head,
.experiment-card-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
}

.training-history-title,
.training-summary-title {
    margin-top: 3px;
    color: var(--body-text-color);
    font-family: 'Poppins', sans-serif;
    font-size: 19px;
    font-weight: 760;
}

.training-history-summary {
    display: grid;
    grid-template-columns: auto auto;
    align-items: baseline;
    gap: 2px 6px;
    padding: 8px 11px;
    border-radius: 13px;
    background: var(--background-fill-secondary);
}

.training-history-summary b {
    color: var(--body-text-color);
    font-size: 13px;
}

.training-history-summary span {
    color: var(--body-text-color-subdued);
    font-size: 10px;
}

.experiment-card-list {
    display: grid;
    gap: 10px;
    margin-top: 13px;
    max-height: 480px;
    overflow-y: auto !important;
    padding-right: 3px;
}

.experiment-card {
    padding: 13px;
    border: 1px solid var(--block-border-color);
    border-radius: 15px;
    background: var(--background-fill-secondary);
}

.experiment-card-best {
    border-color: color-mix(in srgb, #10b981 45%, var(--block-border-color));
    background: color-mix(in srgb, #10b981 7%, var(--block-background-fill));
}

.experiment-id {
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
    font-weight: 850;
    letter-spacing: .9px;
}

.experiment-model {
    margin-top: 3px;
    color: var(--body-text-color);
    font-size: 13.5px;
    font-weight: 750;
    line-height: 1.35;
}

.experiment-best-badge {
    padding: 4px 8px;
    border-radius: 999px;
    background: #10b981;
    color: #fff;
    font-size: 9px;
    font-weight: 850;
    letter-spacing: .65px;
}

.experiment-dataset {
    overflow: hidden;
    margin-top: 7px;
    color: var(--body-text-color-subdued);
    font-size: 11.5px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.experiment-metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.experiment-metrics > div {
    padding: 8px 9px;
    border-radius: 11px;
    background: var(--block-background-fill);
}

.experiment-metrics span,
.training-model-line span,
.training-detail-grid span,
.training-kpi span {
    display: block;
    color: var(--body-text-color-subdued);
    font-size: 9.5px;
    font-weight: 800;
    letter-spacing: .6px;
    text-transform: uppercase;
}

.experiment-metrics b {
    display: block;
    margin-top: 2px;
    color: var(--body-text-color);
    font-size: 16px;
}

.training-summary-card {
    width: 100%;
}

.training-summary-subtitle {
    margin-top: 3px;
    color: var(--body-text-color-subdued);
    font-size: 11.5px;
}

.training-status {
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: .65px;
    text-transform: uppercase;
}

.status-excellent {
    background: color-mix(in srgb, #10b981 13%, var(--block-background-fill));
    color: #059669;
}

.status-good {
    background: color-mix(in srgb, #f59e0b 13%, var(--block-background-fill));
    color: #d97706;
}

.status-review {
    background: color-mix(in srgb, #ef4444 12%, var(--block-background-fill));
    color: #dc2626;
}

.dark .status-excellent { color: #6ee7b7; }
.dark .status-good { color: #fcd34d; }
.dark .status-review { color: #fca5a5; }

.training-model-line {
    margin-top: 13px;
    padding: 11px 12px;
    border-radius: 13px;
    background: var(--background-fill-secondary);
}

.training-model-line b {
    display: block;
    margin-top: 3px;
    color: var(--body-text-color);
    font-size: 13.5px;
}

.training-kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 11px;
}

.training-kpi {
    min-height: 116px;
    padding: 16px;
    border-radius: 17px;
    color: #fff;
    box-shadow: 0 10px 22px rgba(15,23,42,.12);
}

.training-kpi-primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
}

.training-kpi-secondary {
    background: linear-gradient(135deg, #db2777, #f43f5e);
}

.training-kpi span {
    color: rgba(255,255,255,.82);
    font-size: 10px;
}

.training-kpi b {
    display: block;
    margin-top: 12px;
    color: #fff;
    font-size: clamp(30px, 3vw, 42px);
    line-height: 1;
}

.training-insight {
    margin-top: 11px;
    padding: 12px 13px;
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    background: color-mix(in srgb, #6366f1 7%, var(--background-fill-secondary));
    color: var(--body-text-color);
    font-size: 12.5px;
    line-height: 1.55;
}

.training-details {
    margin-top: 11px;
}

.training-details summary {
    cursor: pointer;
    color: color-mix(in srgb, var(--primary-500) 82%, var(--body-text-color));
    font-size: 12.5px;
    font-weight: 720;
}

.training-detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.training-detail-grid > div {
    padding: 9px 10px;
    border-radius: 11px;
    background: var(--background-fill-secondary);
}

.training-detail-grid b {
    display: block;
    margin-top: 3px;
    color: var(--body-text-color);
    font-size: 12px;
}

.advanced-diagnostics {
    margin-top: 12px !important;
}

.deployment-panel {
    gap: 12px !important;
    margin-top: 14px !important;
    padding: 17px !important;
    border: 1px solid color-mix(in srgb, #06b6d4 25%, var(--block-border-color));
    border-radius: 21px;
    background:
        radial-gradient(circle at 105% -40%, rgba(6,182,212,.17), transparent 44%),
        var(--block-background-fill);
    box-shadow: 0 8px 24px rgba(15,23,42,.055);
}

.deployment-panel-copy {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 20px;
}

.deployment-title {
    margin-top: 3px;
    color: var(--body-text-color);
    font-size: 17px;
    font-weight: 760;
}

.deployment-panel-copy p {
    max-width: 560px;
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 12.5px;
    line-height: 1.5;
}

.deployment-controls {
    gap: 10px !important;
}

.advanced-analysis-tabs {
    margin-top: 14px !important;
    padding: 10px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 20px;
    background: var(--block-background-fill);
    box-shadow: 0 8px 24px rgba(15,23,42,.05);
}

/* Slightly larger typography across both modern workspaces. */
.workspace-title,
.results-title {
    font-size: 21px;
}

.workspace-copy {
    font-size: 13.5px;
    line-height: 1.45;
}

.card-heading b {
    font-size: 15px;
}

.card-step,
.workspace-kicker,
.eyebrow {
    font-size: 10px;
}

.card-helper {
    font-size: 12.5px !important;
}

.ui-card label,
.config-card label,
.training-action-card label,
.deployment-panel label,
#advanced-workspace label {
    font-size: 13px !important;
}

.ui-card input,
.ui-card textarea,
.ui-card select,
.config-card input,
.config-card textarea,
.config-card select,
#advanced-workspace input,
#advanced-workspace textarea,
#advanced-workspace select {
    font-size: 13.5px !important;
}

.ui-card button,
.config-card button,
.training-action-card button,
.deployment-panel button,
#advanced-workspace button {
    font-size: 13px !important;
}

.model-profile-copy,
.dataset-ready-meta,
.action-dock-copy small,
.results-hint,
.results-empty small,
.class-pill,
.metric-pill {
    font-size: 10.5px;
}

.model-facts b,
.compact-stat-grid b {
    font-size: 11px;
}

/* =============================================================
   WATER QUALITY ANALYSIS WORKSPACE
   ============================================================= */
.wq-analysis-hero {
    background:
        radial-gradient(circle at 88% -18%, rgba(6,182,212,.22), transparent 36%),
        linear-gradient(135deg,
            color-mix(in srgb, #0891b2 10%, var(--block-background-fill)),
            color-mix(in srgb, #6366f1 7%, var(--block-background-fill)));
}

.wq-analysis-hero-icon {
    background: linear-gradient(135deg, #0891b2, #4f46e5);
    box-shadow: 0 10px 22px rgba(8,145,178,.24);
    font-size: 13px;
    font-weight: 850;
    letter-spacing: .4px;
}

.wq-analysis-top-grid,
.wq-analysis-params-grid,
.wq-analysis-footer-grid {
    align-items: flex-start !important;
    gap: 14px !important;
}

.wq-analysis-card,
.wq-parameter-card,
.wq-analysis-footer-card {
    min-width: 0;
}

.wq-model-choice label {
    line-height: 1.4 !important;
}

.wq-model-note {
    display: grid;
    gap: 3px;
    margin-top: 4px;
    padding: 11px 12px;
    border-left: 4px solid #6366f1;
    border-radius: 11px;
    background: color-mix(in srgb, #6366f1 7%, var(--background-fill-secondary));
}

.wq-model-note b {
    color: var(--body-text-color);
    font-size: 12.5px;
}

.wq-model-note span {
    color: var(--body-text-color-subdued);
    font-size: 11.5px;
    line-height: 1.45;
}

.wq-sheet-row {
    gap: 10px !important;
}

.wq-analysis-section-heading {
    margin-top: 19px;
}

.wq-analysis-run-dock {
    margin-top: 14px !important;
    border-color: color-mix(in srgb, #0891b2 28%, var(--block-border-color));
    background:
        radial-gradient(circle at 105% -60%, rgba(99,102,241,.17), transparent 48%),
        color-mix(in srgb, #0891b2 4%, var(--block-background-fill));
}

.wq-run-orb {
    background: linear-gradient(135deg, #0891b2, #4f46e5) !important;
    font-size: 10px !important;
    font-weight: 850 !important;
    letter-spacing: .4px;
}

.wq-analysis-run-button button,
button.wq-analysis-run-button {
    background: linear-gradient(135deg, #0891b2, #4f46e5) !important;
    box-shadow: 0 9px 22px rgba(8,145,178,.24) !important;
}

.wq-analysis-results-heading {
    margin-top: 21px;
}

.wq-analysis-results-stage {
    gap: 14px !important;
}

.wq-analysis-tabs {
    padding: 10px !important;
    border: 1px solid var(--block-border-color);
    border-radius: 20px;
    background: var(--block-background-fill);
}

.wq-analysis-footer-grid {
    margin-top: 2px !important;
}

.wq-analysis-deployment {
    margin-top: 0 !important;
    min-height: 100%;
}

.wq-analysis-deployment-copy {
    align-items: flex-start;
}

.wq-deployment-buttons {
    gap: 10px !important;
}

#wq-analysis-workspace label,
#wq-analysis-workspace button,
#wq-analysis-workspace input,
#wq-analysis-workspace textarea,
#wq-analysis-workspace select {
    font-size: 13px !important;
}

@media (max-width: 1100px) {
    .advanced-top-grid,
    .advanced-config-grid,
    .advanced-results-grid,
    .wq-analysis-top-grid,
    .wq-analysis-params-grid,
    .wq-analysis-footer-grid {
        flex-wrap: wrap !important;
    }

    .advanced-top-grid > div,
    .advanced-config-grid > div,
    .advanced-results-grid > div,
    .wq-analysis-top-grid > div,
    .wq-analysis-params-grid > div,
    .wq-analysis-footer-grid > div {
        min-width: min(100%, 380px) !important;
    }

    .sticky-training-card,
    .training-setup-card {
        position: static !important;
    }
}

@media (max-width: 720px) {
    .advanced-section-heading,
    .deployment-panel-copy {
        align-items: flex-start;
        flex-direction: column;
    }

    .feature-toggle-row {
        flex-wrap: wrap !important;
    }

    .feature-toggle-row > div {
        min-width: 100% !important;
    }

    .training-kpi-grid,
    .training-detail-grid {
        grid-template-columns: 1fr;
    }

    .deployment-controls {
        flex-direction: column !important;
    }

    .deployment-controls > div {
        width: 100% !important;
    }
}

/* =============================================================
   FINAL STABILITY FIXES — WATER QUALITY TABS + RANGE SLIDERS
   ============================================================= */

/* Keep the application frame at one width while switching tabs.
   A stable vertical scrollbar also prevents the page from nudging
   horizontally when one panel is taller than another. */
html {
    overflow-y: scroll !important;
    scrollbar-gutter: stable !important;
}

body,
.gradio-container {
    min-width: 0 !important;
}

.gradio-container {
    width: min(100%, 1500px) !important;
    max-width: 1500px !important;
    box-sizing: border-box !important;
}

/* Give both Water Quality sub-tabs the same navigation and content frame. */
#wq-subtabs,
.stable-subtabs {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

#wq-subtabs > .tab-nav,
.stable-subtabs > .tab-nav {
    display: flex !important;
    align-items: stretch !important;
    width: 100% !important;
    min-height: 43px !important;
    box-sizing: border-box !important;
}

#wq-subtabs > .tab-nav button,
.stable-subtabs > .tab-nav button {
    flex: 0 0 190px !important;
    width: 190px !important;
    min-width: 190px !important;
    min-height: 42px !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}

#wq-subtabs > .tabitem,
#wq-subtabs > .tab-content,
.stable-subtabs > .tabitem,
.stable-subtabs > .tab-content,
#wq-analysis-workspace {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

/* Modern value-filled range controls.
   Gradio sets --range_progress directly on every Slider input.
   The active rail uses a blue-to-violet gradient up to the handle. */
:root {
    --slider-fill-start: #2563eb;
    --slider-fill-end: #8b5cf6;
    --slider-rail: #d8dee9;
    --slider-rail-border: rgba(71, 85, 105, 0.18);
    --slider-thumb-bg: #ffffff;
    --slider-thumb-border: #6366f1;
}

html.dark,
body.dark,
.dark,
[data-theme="dark"] {
    --slider-fill-start: #38bdf8;
    --slider-fill-end: #a78bfa;
    --slider-rail: #475569;
    --slider-rail-border: rgba(226, 232, 240, 0.16);
    --slider-thumb-bg: #f8fafc;
    --slider-thumb-border: #818cf8;
}

/* All Gradio sliders across every tab and hidden model panel */
.gradio-container input[type="range"],
input[type="range"] {
    --range_progress: 0%;
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 100% !important;
    height: 24px !important;
    min-height: 24px !important;
    margin: 4px 0 6px !important;
    padding: 0 !important;
    border: 0 !important;
    outline: none !important;
    cursor: pointer !important;
    background: transparent !important;
    accent-color: var(--slider-fill-start) !important;
}

/* Chrome, Chromium, Edge and Safari */
.gradio-container input[type="range"]::-webkit-slider-runnable-track,
input[type="range"]::-webkit-slider-runnable-track {
    width: 100% !important;
    height: 9px !important;
    border: 1px solid var(--slider-rail-border) !important;
    border-radius: 999px !important;
    background: linear-gradient(
        90deg,
        var(--slider-fill-start) 0%,
        var(--slider-fill-end) var(--range_progress),
        var(--slider-rail) var(--range_progress),
        var(--slider-rail) 100%
    ) !important;
    box-shadow:
        inset 0 1px 2px rgba(15, 23, 42, 0.12),
        0 1px 1px rgba(255, 255, 255, 0.18) !important;
    transition: filter 0.16s ease, box-shadow 0.16s ease !important;
}

.gradio-container input[type="range"]::-webkit-slider-thumb,
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 20px !important;
    height: 20px !important;
    margin-top: -6px !important;
    border: 3px solid var(--slider-thumb-border) !important;
    border-radius: 50% !important;
    background: var(--slider-thumb-bg) !important;
    box-shadow:
        0 3px 8px rgba(15, 23, 42, 0.28),
        0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

/* Firefox */
.gradio-container input[type="range"]::-moz-range-track,
input[type="range"]::-moz-range-track {
    width: 100% !important;
    height: 9px !important;
    border: 1px solid var(--slider-rail-border) !important;
    border-radius: 999px !important;
    background: var(--slider-rail) !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.12) !important;
}

.gradio-container input[type="range"]::-moz-range-progress,
input[type="range"]::-moz-range-progress {
    height: 9px !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    background: linear-gradient(
        90deg,
        var(--slider-fill-start),
        var(--slider-fill-end)
    ) !important;
}

.gradio-container input[type="range"]::-moz-range-thumb,
input[type="range"]::-moz-range-thumb {
    width: 14px !important;
    height: 14px !important;
    border: 3px solid var(--slider-thumb-border) !important;
    border-radius: 50% !important;
    background: var(--slider-thumb-bg) !important;
    box-shadow:
        0 3px 8px rgba(15, 23, 42, 0.28),
        0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.gradio-container input[type="range"]:hover::-webkit-slider-runnable-track,
input[type="range"]:hover::-webkit-slider-runnable-track {
    filter: saturate(1.12) brightness(1.03) !important;
    box-shadow:
        inset 0 1px 2px rgba(15, 23, 42, 0.13),
        0 0 0 3px rgba(99, 102, 241, 0.09) !important;
}

.gradio-container input[type="range"]:hover::-webkit-slider-thumb,
.gradio-container input[type="range"]:focus-visible::-webkit-slider-thumb,
input[type="range"]:hover::-webkit-slider-thumb,
input[type="range"]:focus-visible::-webkit-slider-thumb {
    transform: scale(1.10) !important;
    box-shadow:
        0 4px 11px rgba(15, 23, 42, 0.30),
        0 0 0 6px rgba(99, 102, 241, 0.18) !important;
}

.gradio-container input[type="range"]:hover::-moz-range-thumb,
.gradio-container input[type="range"]:focus-visible::-moz-range-thumb,
input[type="range"]:hover::-moz-range-thumb,
input[type="range"]:focus-visible::-moz-range-thumb {
    transform: scale(1.10) !important;
    box-shadow:
        0 4px 11px rgba(15, 23, 42, 0.30),
        0 0 0 6px rgba(99, 102, 241, 0.18) !important;
}

.gradio-container input[type="range"]:focus-visible,
input[type="range"]:focus-visible {
    outline: none !important;
}

.gradio-container input[type="range"]:disabled,
input[type="range"]:disabled {
    cursor: not-allowed !important;
    opacity: 0.58 !important;
    filter: grayscale(0.25) !important;
}

@media (max-width: 720px) {
    #wq-subtabs > .tab-nav button,
    .stable-subtabs > .tab-nav button {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0 !important;
        padding-inline: 10px !important;
    }
}
"""
