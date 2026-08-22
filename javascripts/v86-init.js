(function () {
  if (typeof document$ === "undefined") return;

  var emulator = null;
  var libPromise = null;
  var keydownHandlerRemoved = false;

  function loadLib() {
    if (libPromise) return libPromise;
    libPromise = new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = "/javascripts/v86/libv86.js";
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
    return libPromise;
  }

  function removeMaterialKeydown() {
    if (keydownHandlerRemoved) return;
    var list = window.__v86KeydownListeners || [];
    if (list.length === 0) return;
    var first = list.shift();
    window.removeEventListener("keydown", first.listener, first.options);
    keydownHandlerRemoved = true;
  }

  document$.subscribe(function () {
    if (emulator) {
      try { emulator.destroy(); } catch (e) {}
      emulator = null;
    }

    var container = document.getElementById("screen_container");
    if (!container) return;

    loadLib().then(function () {
      removeMaterialKeydown();
      emulator = new V86({
        wasm_path: "/javascripts/v86/v86.wasm",
        memory_size: 512 * 1024 * 1024,
        vga_memory_size: 8 * 1024 * 1024,
        screen_container: container,
        bios: { url: "/javascripts/v86/seabios.bin" },
        vga_bios: { url: "/javascripts/v86/vgabios.bin" },
        bzimage: { url: "/javascripts/v86/buildroot-bzimage68.bin" },
        autostart: true,
      });
    });
  });
})();
