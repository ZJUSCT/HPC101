var MATHJAX_CONFIG = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// 此处深拷贝原因：MathJax 加载后向 window.MathJax 上挂内部属性，复用会导致第二次加载跳过初始化
function mathjaxConfig() {
  return JSON.parse(JSON.stringify(MATHJAX_CONFIG));
}

window.MathJax = mathjaxConfig();

// navigation.instant 下每次切页重新加载 MathJax，重建菜单状态
document$.subscribe(function () {
  window.MathJax = mathjaxConfig();
  var existing = document.querySelector('script[src*="mathjax"]');
  if (existing) existing.remove();
  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mathjax/es5/tex-mml-chtml.js";
  document.head.appendChild(script);
});
