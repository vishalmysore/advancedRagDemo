/* coi-serviceworker v0.1.7 — MIT License — https://github.com/gzuidhof/coi-serviceworker
 * Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers so that
 * SharedArrayBuffer (required by WebLLM / WebGPU) is available on hosts like GitHub Pages
 * that cannot set HTTP headers directly.
 */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", function (event) {
  if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") return;

  // Only inject COEP/COOP headers for same-origin responses.
  // Cross-origin CDN fetches (e.g. pdf.js, model weights) must pass through
  // unmodified; wrapping them causes network errors under require-corp COEP.
  const isSameOrigin = new URL(event.request.url).origin === self.location.origin;
  if (!isSameOrigin) return;

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
      .catch((e) => {
        console.error(e);
        return new Response(null, { status: 503, statusText: "Service Unavailable" });
      })
  );
});
