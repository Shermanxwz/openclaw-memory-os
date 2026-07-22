import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { registerMemoryCapability } from "openclaw/plugin-sdk/memory-core";
import { getMemorySearchManager as getBuiltinMemorySearchManager } from "openclaw/plugin-sdk/memory-core-engine-runtime";
import { MemoryOsSearchManager, normalizeConfig } from "./adapter.js";

function pluginConfig(api) {
  const raw = api?.config ?? api?.runtime?.pluginConfig ?? {};
  return normalizeConfig(raw);
}

export default definePluginEntry({
  id: "openclaw-memory-os",
  name: "OpenClaw Memory OS Adapter",
  description: "Search OpenClaw Memory OS first, then fall back to built-in OpenClaw memory search.",
  kind: "memory",
  register(api) {
    const cfg = pluginConfig(api);
    registerMemoryCapability("openclaw-memory-os", {
      runtime: {
        async getMemorySearchManager(params) {
          let fallbackManager = null;
          let fallbackError = null;
          if (cfg.fallback) {
            try {
              const result = await getBuiltinMemorySearchManager(params);
              fallbackManager = result?.manager ?? null;
              fallbackError = result?.error ?? null;
            } catch (error) {
              fallbackError = error instanceof Error ? error.message : String(error);
            }
          }
          const manager = new MemoryOsSearchManager({
            config: cfg,
            fallbackManager,
            logger: api?.logger
          });
          if (fallbackError) manager.lastError = `fallback init: ${fallbackError}`;
          return { manager };
        },
        resolveMemoryBackendConfig() {
          return { backend: "builtin" };
        },
        async closeMemorySearchManager() {},
        async closeAllMemorySearchManagers() {}
      }
    });
  }
});

export { MemoryOsSearchManager, normalizeConfig } from "./adapter.js";
