/* Rewrite explicitly opted-in examples to a reader's preferred Gemma 4 model. */
(() => {
  "use strict";

  const STORAGE_KEY = "gemma-model";
  const QUERY_KEY = "model";
  const models = window.GEMMA_MODELS || {};
  const index = [];

  for (const [id, model] of Object.entries(models)) {
    for (const [engine, value] of Object.entries(model.ids || {})) {
      index.push({ text: value, engine, model: id });
    }
  }
  index.sort((a, b) => b.text.length - a.text.length);

  const blocks = () => document.querySelectorAll(".model-swap code, code.model-swap");
  const displayName = (id) => models[id]?.fullName || models[id]?.name || id;

  const readStored = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  };

  const writeStored = (value) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
    } catch {
      // Storage may be disabled; the current page still works.
    }
  };

  const readUrl = () => new URL(window.location.href).searchParams.get(QUERY_KEY);

  const writeUrl = (value) => {
    const url = new URL(window.location.href);
    url.searchParams.set(QUERY_KEY, value);
    window.history.replaceState(window.history.state, "", url);
  };

  const enginesInUse = () => {
    const found = new Set();
    for (const block of blocks()) {
      for (const entry of index) {
        if (block.textContent.includes(entry.text)) found.add(entry.engine);
      }
    }
    return found;
  };

  const candidates = (requires) => {
    const engines = [...enginesInUse()];
    if (!engines.length) return [];
    return Object.keys(models).filter((id) => {
      const model = models[id];
      if (requires && !(model.inputs || []).includes(requires)) return false;
      return engines.every((engine) => (model.ids || {})[engine]);
    });
  };

  const detect = () => {
    for (const block of blocks()) {
      const hit = index.find((entry) => block.textContent.includes(entry.text));
      if (hit) return hit.model;
    }
    return null;
  };

  const rewriteNode = (node, target) => {
    const original = node.__gemmaOriginal ?? node.nodeValue;
    let next = original;
    for (const entry of index) {
      if (!next.includes(entry.text)) continue;
      const replacement = models[target]?.ids?.[entry.engine];
      if (replacement) next = next.split(entry.text).join(replacement);
    }
    if (next === node.nodeValue) return;
    node.__gemmaOriginal = original;
    node.nodeValue = next;
  };

  const apply = (target) => {
    if (!models[target]) return;
    for (const block of blocks()) {
      const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) rewriteNode(node, target);
    }
  };

  const closePicker = (picker, returnFocus = false) => {
    const trigger = picker.querySelector(".page-context__picker-trigger");
    const menu = picker.querySelector(".page-context__picker-menu");
    trigger.setAttribute("aria-expanded", "false");
    menu.hidden = true;
    if (returnFocus) trigger.focus();
  };

  const updateColab = (target) => {
    const colabLink = document.querySelector(".page-context__colab");
    if (!colabLink) return;
    const slug = `${target.replace("/", "-")}`;
    colabLink.href = colabLink.href.replace(/-gemma-4-[a-z0-9-]+\.ipynb/, `-${slug}.ipynb`);
  };

  const choose = (picker, target) => {
    const trigger = picker.querySelector(".page-context__picker-trigger");
    trigger.querySelector(".page-context__picker-name").textContent = displayName(target);
    trigger.dataset.gemmaModel = target;
    trigger.setAttribute("aria-label", `Model used by this page's examples: ${displayName(target)}`);
    for (const option of picker.querySelectorAll("[role=option]")) {
      option.setAttribute("aria-selected", String(option.dataset.value === target));
    }
    updateColab(target);
  };

  const buildControl = (strip, available, initial, unavailable) => {
    const item = document.createElement("span");
    item.className = "page-context__item";

    const label = document.createElement("span");
    label.className = "page-context__label";
    label.textContent = "Examples";

    const picker = document.createElement("span");
    picker.className = "page-context__picker";

    const menuId = `gemma-model-menu-${Math.random().toString(36).slice(2, 9)}`;
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "page-context__picker-trigger";
    trigger.dataset.gemmaModel = initial;
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", menuId);
    trigger.innerHTML = '<span class="page-context__picker-name"></span>';

    const menu = document.createElement("span");
    menu.className = "page-context__picker-menu";
    menu.id = menuId;
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", "Model used by this page's examples");
    menu.hidden = true;

    if (unavailable && models[unavailable]) {
      const notice = document.createElement("span");
      notice.className = "page-context__picker-notice";
      notice.textContent = `${displayName(unavailable)} isn't available for this page.`;
      menu.appendChild(notice);
    }

    for (const id of available) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "page-context__picker-option";
      option.dataset.value = id;
      option.setAttribute("role", "option");
      option.innerHTML =
        `<span class="page-context__picker-option-name">${displayName(id)}</span>` +
        `<span class="page-context__picker-option-meta">${models[id].platform} · ${models[id].context} context</span>`;
      option.addEventListener("click", () => {
        choose(picker, id);
        writeStored(id);
        writeUrl(id);
        apply(id);
        closePicker(picker, true);
      });
      option.addEventListener("keydown", (event) => {
        const options = [...menu.querySelectorAll("[role=option]")];
        const current = options.indexOf(option);
        const destinations = {
          ArrowDown: options[(current + 1) % options.length],
          ArrowUp: options[(current - 1 + options.length) % options.length],
          Home: options[0],
          End: options.at(-1),
        };
        if (destinations[event.key]) {
          event.preventDefault();
          destinations[event.key].focus();
        } else if (event.key === "Escape") {
          event.preventDefault();
          closePicker(picker, true);
        }
      });
      menu.appendChild(option);
    }

    trigger.addEventListener("click", () => {
      const opening = menu.hidden;
      document.querySelectorAll(".page-context__picker").forEach((other) => closePicker(other));
      menu.hidden = !opening;
      trigger.setAttribute("aria-expanded", String(opening));
    });
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closePicker(picker, true);
        return;
      }
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      const options = [...menu.querySelectorAll("[role=option]")];
      (event.key === "ArrowDown" ? options[0] : options.at(-1))?.focus();
    });

    picker.append(trigger, menu);
    item.append(label, picker);
    const colabItem = strip.querySelector(".page-context__item--colab");
    if (colabItem) strip.insertBefore(item, colabItem);
    else strip.appendChild(item);
    choose(picker, initial);
  };

  const init = () => {
    const strip = document.querySelector(".page-context");
    const requested = readUrl() || readStored();

    if (!strip || document.querySelector("[data-gemma-model]")) return;

    const available = candidates(strip.dataset.requires || "");
    if (available.length < 2) {
      if (requested && available.includes(requested)) {
        apply(requested);
        updateColab(requested);
      }
      return;
    }

    const detected = detect();
    const initial = available.includes(requested)
      ? requested
      : available.includes(detected)
        ? detected
        : available[0];

    buildControl(strip, available, initial, requested && !available.includes(requested) ? requested : null);
    if (requested && available.includes(requested)) apply(requested);
  };

  document.addEventListener("click", (event) => {
    for (const picker of document.querySelectorAll(".page-context__picker")) {
      if (!picker.contains(event.target)) closePicker(picker);
    }
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
  if (window.document$?.subscribe) window.document$.subscribe(init);
})();
