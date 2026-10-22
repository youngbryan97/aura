import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const source = fs.readFileSync(`${process.argv[2]}/interface/static/aura.js`, 'utf8');
const start = source.indexOf('async function reconcileAutoListenFromSettings(');
const end = source.indexOf('async function hydrateRuntimeSettings(', start);
assert.ok(start >= 0 && end > start);
const calls = [];
let active = true;
let ambient = false;
const sandbox = {
  runtimeSettingsState: {hydrated:true, values:{'voice.input_enabled':true, 'voice.auto_listen':true}},
  state: {voiceSummary:{server_capture:true, listening:true}},
  window: {AuraVoiceMode:{
    isActive:()=>active, isAmbient:()=>ambient,
    setAmbient:async on=>{calls.push(['ambient',on]);active=on;return on;},
  }},
  toggleVoice:async on=>{calls.push(['voice',on]);active=on;return true;},
};
vm.createContext(sandbox);
vm.runInContext(source.slice(start,end), sandbox);
await sandbox.reconcileAutoListenFromSettings();
assert.equal(active,true);
assert.deepEqual(calls,[]);
sandbox.runtimeSettingsState.values['voice.auto_listen']=false;
await sandbox.reconcileAutoListenFromSettings();
assert.equal(active,true);
assert.deepEqual(calls,[]);
sandbox.runtimeSettingsState.values['voice.input_enabled']=false;
await sandbox.reconcileAutoListenFromSettings();
assert.equal(active,false);
assert.deepEqual(calls,[['voice',false]]);
active=true;ambient=true;calls.length=0;
sandbox.runtimeSettingsState.values['voice.input_enabled']=true;
await sandbox.reconcileAutoListenFromSettings();
assert.equal(active,false);
assert.deepEqual(calls,[['ambient',false]]);
console.log('Focused voice survives settings refresh; microphone disable remains authoritative.');
