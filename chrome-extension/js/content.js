/* =========================================================================
 * PHASE 5 — Content script
 * Injecté dans la page youtube.com/watch, il extrait les commentaires visibles.
 *
 * Difficulté principale : YouTube charge les commentaires en LAZY LOADING.
 * À l'ouverture, le DOM n'en contient aucun ; ils n'apparaissent qu'au scroll.
 * Stratégie retenue :
 *   1. scroller par paliers jusqu'à la section #comments,
 *   2. attendre entre chaque palier que de nouveaux noeuds apparaissent,
 *   3. s'arrêter dès qu'on atteint la cible ou qu'aucun nouveau commentaire
 *      n'arrive après 3 tentatives (fin de liste).
 *
 * Les réponses (threads) sont distinguées des commentaires racine grâce au
 * conteneur #replies : on les marque `isReply` pour pouvoir les filtrer.
 * ========================================================================= */

(() => {
  "use strict";

  const SELECTOR_COMMENT = "ytd-comment-thread-renderer #content-text";
  const SELECTOR_SECTION = "ytd-comments#comments";

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /** Lit les commentaires actuellement présents dans le DOM. */
  function readComments() {
    const nodes = document.querySelectorAll(SELECTOR_COMMENT);
    const out = [];
    nodes.forEach((node) => {
      const text = (node.innerText || "").trim().replace(/\s+/g, " ");
      if (!text) return;
      const thread = node.closest("ytd-comment-thread-renderer");
      const isReply = !!node.closest("#replies");
      const authorEl = thread ? thread.querySelector("#author-text") : null;
      out.push({
        text,
        author: authorEl ? authorEl.innerText.trim() : "",
        isReply,
      });
    });
    return out;
  }

  /** Scrolle jusqu'à obtenir `target` commentaires (ou la fin de la liste). */
  async function loadComments(target = 100) {
    const section = document.querySelector(SELECTOR_SECTION);
    if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
    await sleep(1200);

    let previous = 0;
    let stalled = 0;

    while (stalled < 3) {
      const current = document.querySelectorAll(SELECTOR_COMMENT).length;
      if (current >= target) break;
      if (current === previous) stalled += 1;
      else stalled = 0;

      previous = current;
      window.scrollBy({ top: window.innerHeight * 1.5, behavior: "smooth" });
      await sleep(900);
    }
    return readComments().slice(0, target);
  }

  /** Métadonnées de la vidéo courante, pour l'en-tête du popup. */
  function videoInfo() {
    const titleEl =
      document.querySelector("h1.ytd-watch-metadata yt-formatted-string") ||
      document.querySelector("h1.title yt-formatted-string");
    const channelEl = document.querySelector("#channel-name #text a, ytd-channel-name a");
    return {
      title: titleEl ? titleEl.innerText.trim() : document.title,
      channel: channelEl ? channelEl.innerText.trim() : "",
      url: location.href,
    };
  }

  /* --- Canal de communication avec le popup ----------------------------- */
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.action === "ping") {
      sendResponse({ ok: true });
      return false;
    }
    if (msg.action === "extractComments") {
      loadComments(msg.limit || 100)
        .then((comments) =>
          sendResponse({ ok: true, comments, video: videoInfo() })
        )
        .catch((err) => sendResponse({ ok: false, error: String(err) }));
      return true; // réponse asynchrone
    }
    return false;
  });

  console.log("[YT-Sentiment] content script prêt");
})();
