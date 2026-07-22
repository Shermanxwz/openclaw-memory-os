# OpenClaw Memory OS Adapter Plugin

`plugins/openclaw-memory-os-adapter/` is a native OpenClaw **exclusive memory plugin**.
When selected with `plugins.slots.memory = "openclaw-memory-os"`, OpenClaw's built-in
`memory_search` path asks Memory OS first and falls back to the built-in OpenClaw
memory-core manager when Memory OS is unavailable or returns no hits.

This is the route-B integration: Memory OS becomes the active memory runtime, not
just an optional `corpus=all` supplement.

## Why this exists

Memory OS has richer governance metadata than vanilla semantic search:

- `status=active` by default
- superseded / expired suppression
- tier and importance ranking
- recall explanations
- feedback and audit hooks

The adapter lets OpenClaw use that smarter recall layer without patching the
installed OpenClaw package under `/usr/lib/node_modules/openclaw`.

## Files

```text
plugins/openclaw-memory-os-adapter/
├── openclaw.plugin.json
├── package.json
├── src/
│   ├── adapter.js
│   └── index.js
└── test/
    └── adapter.test.js
```

## Runtime behavior

```text
OpenClaw memory_search
  ↓
active memory slot: openclaw-memory-os
  ↓
MemoryOsSearchManager.search(query)
  ↓
POST {baseUrl}/api/recall-test
  ↓
if hits found: return Memory OS-ranked hits
if failed/empty and fallback=true: call built-in memory-core manager
```

The fallback path uses OpenClaw's SDK export:

```js
import { getMemorySearchManager as getBuiltinMemorySearchManager } from "openclaw/plugin-sdk/memory-core-engine-runtime";
```

## Config

Example `openclaw.json` fragment:

```json5
{
  plugins: {
    load: {
      paths: [
        "/path/to/openclaw-memory-os/plugins/openclaw-memory-os-adapter"
      ]
    },
    slots: {
      memory: "openclaw-memory-os"
    },
    entries: {
      "openclaw-memory-os": {
        enabled: true,
        config: {
          baseUrl: "http://127.0.0.1:7788",
          token: "${MEMORY_OS_TOKEN}",
          timeoutMs: 2500,
          fallback: true
        }
      }
    }
  }
}
```

The plugin also reads environment defaults:

- `MEMORY_OS_URL` → default `http://127.0.0.1:7788`
- `MEMORY_OS_TOKEN` → optional bearer token

## Safety

- The adapter does not delete or mutate memories.
- On Memory OS outage, timeout, or HTTP error, it falls back to native OpenClaw
  memory search when `fallback=true`.
- The plugin lives in this repo and is loaded through OpenClaw's plugin system;
  it does not modify OpenClaw's installed `dist/` files.

## Tests

```bash
node --test plugins/openclaw-memory-os-adapter/test/*.test.js
```

The plugin imports OpenClaw SDK subpaths as peer dependencies. OpenClaw resolves
those when loading the plugin. For standalone local import checks outside the
OpenClaw runtime, link or install OpenClaw into the plugin's `node_modules` first.
For example, in a disposable dev check:

```bash
mkdir -p plugins/openclaw-memory-os-adapter/node_modules
ln -s /usr/lib/node_modules/openclaw plugins/openclaw-memory-os-adapter/node_modules/openclaw
node -e "import('./plugins/openclaw-memory-os-adapter/src/index.js').then(m => console.log(typeof m.default))"
rm -rf plugins/openclaw-memory-os-adapter/node_modules
```

Current adapter tests cover:

- config defaults
- recall payload construction
- Memory OS hit mapping into OpenClaw memory result shape
- HTTP call URL/header/body behavior
- fallback when Memory OS fails
