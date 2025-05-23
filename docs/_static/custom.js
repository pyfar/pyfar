/**
 * Cleans up the third-level TOC entries in the secondary sidebar.
 *
 * This function is used to simplify the attribute names in secondary sidebar
 * entries by removing the class name prefix.
 *
 * Example:
 *   "Class.attribute" → "attribute"
 *
 * @function cleanTocAttributeEntries
 * @returns {void}
 */
function cleanTocAttributeEntries() {
    // Only target spans inside the secondary sidebar
    const sidebarToc = document.querySelector(".bd-sidebar-secondary");
    if (!sidebarToc) return;

    // Select spans only inside <li class="toc-h3 nav-item toc-entry">
    const tocSpans = sidebarToc.querySelectorAll("li.toc-h3.nav-item.toc-entry .pre");

    tocSpans.forEach(span => {
        const original = span.textContent;
        const firstDotIndex = original.indexOf(".");
        span.textContent = original.slice(firstDotIndex + 1); // Keep everything after the first dot
    });
};

// Run after html document has been parsed
document.addEventListener("DOMContentLoaded", () => {
    cleanTocAttributeEntries();
});
