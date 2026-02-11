// MathJax configuration for LaTeX rendering in documentation

window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        tags: 'ams',
        tagSide: 'right',
        tagIndent: '.8em',
        packages: { '[+]': ['ams', 'newcommand', 'configmacros'] }
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    },
    svg: {
        fontCache: 'global'
    }
};

// Re-render MathJax when page content changes (for SPA navigation)
document$.subscribe(() => {
    MathJax.typesetPromise()
})
