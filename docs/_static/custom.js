document.addEventListener("DOMContentLoaded", () => {
    // Only target spans inside the page ToC
    const sidebarToc = document.querySelector(".bd-sidebar-secondary");
    if (!sidebarToc) return;

    // Select spans only inside <li class="toc-h3 nav-item toc-entry">
    const tocSpans = sidebarToc.querySelectorAll("li.toc-h3.nav-item.toc-entry .pre");

    tocSpans.forEach(span => {
        const original = span.textContent;
        const firstDotIndex = original.indexOf(".");
        span.textContent = original.slice(firstDotIndex + 1); // Keep everything after the first dot
    });
});
