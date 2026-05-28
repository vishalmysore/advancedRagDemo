/* coi-serviceworker v0.1.7 — MIT License — https://github.com/gzuidhof/coi-serviceworker
 * Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers so that
 * SharedArrayBuffer (required by WebLLM / WebGPU) is available on hosts like GitHub Pages
 * that cannot set HTTP headers directly.
 */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", function (event) {
  if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const newHeaders = new Headers(response.headers);
        newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
        newHeaders.set("Cross-Origin-Embedder-Policy", "require-corp");
        newHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      })
      .catch((e) => console.error(e))
  );
});
