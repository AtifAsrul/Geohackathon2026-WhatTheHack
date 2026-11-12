/* Run with: node tests/test_well_import_ui.js. No browser or real data needed. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8");

async function main() {
  const backend = { volumes: Array(10).fill({}), wells: Array(6).fill({}), interp: { ready: false } };
  let polls = 0, refreshes = 0;
  const context = {
    S: { vols: Array(10).fill({}), wells: [], status: { stage: "wells" }, interp: { ready: false } },
    setTimeout: (f) => f(), showProgress: () => {}, addBot: () => {},
    getJSON: async (url) => {
      if (url === "/api/state") return backend;
      return ++polls === 1 ? { stage: "grid", state: "running" } : { state: "done" };
    },
    loadState: async () => { refreshes++; context.S.wells = backend.wells; },
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf("let polling = false;"), source.indexOf("// import dialog")), context);
  await context.pollStatus();
  assert.equal(polls, 1, "The well counter must refresh before the interpretation finishes");
  assert.equal(refreshes, 1);
  assert.equal(context.S.wells.length, 6);

  const fields = { "imp-path": {}, "imp-wells": { value: "old guess" }, "imp-list": {}, "imp-found": {}, "imp-use": {} };
  const browseContext = {
    $: (id) => fields[id], esc: String, scanWells: () => {},
    document: { querySelectorAll: () => [] },
    getJSON: async () => ({ path: "/survey/stack", parent: "/survey", wells_guess: "", dirs: [], files: [], segy_found: 1, las_found: 0 }),
  };
  vm.createContext(browseContext);
  vm.runInContext(source.slice(source.indexOf("async function browse(path)"), source.indexOf('$("btn-import").onclick')), browseContext);
  await browseContext.browse("/survey/stack");
  assert.equal(fields["imp-wells"].value, "", "A seismic-only folder must not become the well input");
  browseContext.browse.wellsEdited = true;
  fields["imp-wells"].value = "/explicit/wells";
  await browseContext.browse("/survey/stack");
  assert.equal(fields["imp-wells"].value, "/explicit/wells", "Preserve an explicit well-folder selection");
  console.log("Well import UI regressions passed (live counter, empty guess, explicit selection).");
}
main().catch((e) => { console.error(e); process.exitCode = 1; });
