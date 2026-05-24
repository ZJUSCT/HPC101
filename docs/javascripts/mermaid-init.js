document$.subscribe(function () {
  if (typeof mermaid !== "undefined") {
    mermaid.run({ querySelector: ".mermaid" });
  }
});
