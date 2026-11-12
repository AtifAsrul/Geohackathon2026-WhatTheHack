/* Real slider handlers, exercised without a browser or seismic data. */
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const assert = require("node:assert/strict");
const source = fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8");
const fields = Object.fromEntries(["fault-cutoff", "fault-cutoff-value", "fault-cutoff-status", "btn-apply-faults"].map(id => [id, {}]));
let previews = 0, posts = [];
const c = {
  S: {faultMax: 1, faultSaved: 1, faultDirty: false, interp: {ready: true, fault_settings: {method: "FaultSSL Precision First"}}, status: {}, layers: {}},
  $: id => fields[id], document: {querySelector: () => ({classList: {add: () => {}}})},
  clearTimeout: () => {}, setTimeout: fn => fn(), updateSection: () => previews++, updateMap: () => previews++,
  note: message => { throw new Error(message); }, showProgress: () => {}, pollStatus: () => {},
  getJSON: async (url, options) => {
    if (options) { posts.push({url, body: JSON.parse(options.body)}); return {ok: true}; }
    return {state: "running"};
  },
};
vm.createContext(c);
vm.runInContext(source.slice(source.indexOf("function syncFaultControls()"), source.indexOf("async function loadState(")), c);
(async () => {
  fields["fault-cutoff"].oninput({target: {value: "0.5"}});
  assert.equal(c.S.faultMax, 0.5);
  assert.equal(fields["fault-cutoff-value"].textContent, "0.50");
  assert.equal(c.S.faultDirty, true);
  assert.equal(c.S.layers.faults, true);
  assert.equal(previews, 2);
  assert.equal(posts.length, 0, "Dragging must not modify stored results");
  await fields["btn-apply-faults"].onclick();
  assert.deepEqual(posts, [{url: "/api/fault_settings", body: {coherence_max: 0.5}}]);
  assert.equal(fields["btn-apply-faults"].disabled, true);
  c.S.status = {};
  fields["fault-cutoff"].oninput({target: {value: "1"}});
  assert.equal(fields["fault-cutoff-value"].textContent, "off");
  assert.equal(c.S.faultDirty, false);
  console.log("Fault slider UI checks passed: preview only, explicit apply, busy state and off.");
})().catch(error => { console.error(error); process.exitCode = 1; });
