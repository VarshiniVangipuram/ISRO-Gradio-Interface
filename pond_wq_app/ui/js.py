DROPDOWN_JS = """
() => {
    // ─── Dropdown Portal Fix ───────────────────────────────────────────────
    // Gradio 4.x renders the listbox inside a Svelte component that may be
    // clipped by overflow:hidden ancestors. We fix this by switching the
    // listbox to position:fixed and computing its position from the
    // bounding-rect of the input wrapper, so it always floats above everything.

    function positionDropdown(lb) {
        // ── Step 1: Walk up the DOM from the listbox and free every
        //    ancestor (up to the dropdown root) from overflow:hidden /
        //    fixed-height constraints that Svelte scoped CSS may set.
        const dropRoot = lb.closest('[data-testid="dropdown"]')
                      || lb.closest('.gr-dropdown')
                      || lb.closest('.wrap');
        let el = lb.parentElement;
        while (el && el !== dropRoot && el !== document.body) {
            el.style.setProperty('overflow', 'visible', 'important');
            el.style.setProperty('height', 'auto', 'important');
            el.style.setProperty('max-height', 'none', 'important');
            el = el.parentElement;
        }

        // ── Step 2: Find the anchor whose bounding-rect we use for positioning.
        const anchor = lb.closest('.wrap')
                    || lb.closest('[data-testid="dropdown"]')
                    || lb.parentElement;
        if (!anchor) return;

        const rect  = anchor.getBoundingClientRect();
        const viewH = window.innerHeight;
        const maxH  = 280;   // use fixed desired height, NOT the (possibly clipped) offsetHeight

        // Open downward unless there is not enough space below
        const openDown = (rect.bottom + maxH + 8) <= viewH || rect.top < maxH;
        const topPx    = openDown ? rect.bottom + 4 : rect.top - maxH - 4;

        // ── Step 3: Pin the listbox itself with fixed positioning + scroll.
        lb.style.setProperty('position',   'fixed',   'important');
        lb.style.setProperty('z-index',    '999999',  'important');
        lb.style.setProperty('top',        topPx + 'px', 'important');
        lb.style.setProperty('left',       rect.left + 'px', 'important');
        lb.style.setProperty('width',      rect.width + 'px', 'important');
        lb.style.setProperty('min-width',  rect.width + 'px', 'important');
        lb.style.setProperty('max-height', maxH + 'px', 'important');
        lb.style.setProperty('height',     'auto',    'important');
        lb.style.setProperty('overflow-y', 'auto',    'important');
        lb.style.setProperty('overflow-x', 'hidden',  'important');
        lb.style.setProperty('display',    'block',   'important');
    }

    const repositionAll = () => {
        document.querySelectorAll('ul[role="listbox"]').forEach(lb => {
            if (lb.offsetWidth > 0 || lb.offsetHeight > 0) {
                positionDropdown(lb);
            }
        });
    };

    // 1. Mutation Observer for instant positioning when Svelte mounts the dropdown
    const mo = new MutationObserver(mutations => {
        let found = false;
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.matches('ul[role="listbox"]')) {
                    positionDropdown(node);
                    found = true;
                } else {
                    const lb = node.querySelector && node.querySelector('ul[role="listbox"]');
                    if (lb) {
                        positionDropdown(lb);
                        found = true;
                    }
                }
            }
        }
        if (found) {
            // Re-run after a tiny delay in case layout shifts sizes
            setTimeout(repositionAll, 30);
        }
    });
    mo.observe(document.body, { childList: true, subtree: true });

    // 2. Global event listeners for user interactions (click / focus)
    document.addEventListener('click', () => setTimeout(repositionAll, 20), true);
    document.addEventListener('focusin', () => setTimeout(repositionAll, 20), true);

    // 3. Scroll and resize listeners
    window.addEventListener('scroll', repositionAll, true);
    window.addEventListener('resize', repositionAll);

    // 4. Fallback interval to ensure dropdowns are always corrected
    setInterval(repositionAll, 150);

    // ─── Range Slider Progress Synchronization ───────────────────────────
    // Gradio's Slider component paints its active rail using --range_progress.
    // Keep that exact property synchronized for initial mount, hidden panels,
    // Undo restores, reset buttons, number-field edits, and programmatic updates.
    function syncRange(range) {
        if (!range || range.type !== 'range') return;

        const min = Number.isFinite(parseFloat(range.min))
            ? parseFloat(range.min)
            : 0;
        const max = Number.isFinite(parseFloat(range.max))
            ? parseFloat(range.max)
            : 100;
        const value = Number.isFinite(parseFloat(range.value))
            ? parseFloat(range.value)
            : min;

        const span = Math.max(max - min, Number.EPSILON);
        const pct = Math.max(
            0,
            Math.min(100, ((value - min) / span) * 100)
        );

        // Use Gradio's actual variable name: underscore, not hyphen.
        range.style.setProperty(
            '--range_progress',
            pct.toFixed(4) + '%'
        );
    }

    function syncAllRanges(root = document) {
        if (
            root.matches &&
            root.matches('input[type="range"]')
        ) {
            syncRange(root);
        }

        if (root.querySelectorAll) {
            root.querySelectorAll(
                'input[type="range"]'
            ).forEach(syncRange);
        }
    }

    document.addEventListener(
        'input',
        event => {
            if (
                event.target &&
                event.target.matches('input[type="range"]')
            ) {
                syncRange(event.target);
            }

            // A Slider's number field can also change its range value.
            if (
                event.target &&
                event.target.matches('input[type="number"]')
            ) {
                const wrap = event.target.closest('.wrap');
                const range = wrap?.querySelector(
                    'input[type="range"]'
                );
                if (range) {
                    requestAnimationFrame(
                        () => syncRange(range)
                    );
                }
            }
        },
        true
    );

    document.addEventListener(
        'change',
        event => {
            if (
                event.target &&
                event.target.matches(
                    'input[type="range"], input[type="number"]'
                )
            ) {
                const wrap = event.target.closest('.wrap');
                const range = event.target.matches(
                    'input[type="range"]'
                )
                    ? event.target
                    : wrap?.querySelector(
                        'input[type="range"]'
                    );

                if (range) {
                    requestAnimationFrame(
                        () => syncRange(range)
                    );
                }
            }
        },
        true
    );

    document.addEventListener(
        'click',
        event => {
            // Covers Gradio's reset button.
            if (
                event.target?.closest?.(
                    '[data-testid="reset-button"]'
                )
            ) {
                const wrap = event.target.closest('.wrap');
                const range = wrap?.querySelector(
                    'input[type="range"]'
                );
                if (range) {
                    requestAnimationFrame(
                        () => syncRange(range)
                    );
                    setTimeout(
                        () => syncRange(range),
                        30
                    );
                }
            }
        },
        true
    );

    const sliderObserver = new MutationObserver(
        mutations => {
            for (const mutation of mutations) {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) {
                        syncAllRanges(node);
                    }
                });

                if (
                    mutation.type === 'attributes' &&
                    mutation.target.matches?.(
                        'input[type="range"]'
                    )
                ) {
                    syncRange(mutation.target);
                }
            }
        }
    );

    sliderObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: [
            'value',
            'min',
            'max',
            'style',
            'class'
        ]
    });

    // Cover first render, delayed Gradio hydration, tab switches,
    // model-panel visibility changes, and restored experiments.
    syncAllRanges();
    requestAnimationFrame(
        () => syncAllRanges()
    );
    setTimeout(
        () => syncAllRanges(),
        50
    );
    setTimeout(
        () => syncAllRanges(),
        180
    );
    setTimeout(
        () => syncAllRanges(),
        500
    );
    setInterval(
        () => syncAllRanges(),
        500
    );
}
"""
