import { sites } from "@openai/sites-vite-plugin";
import vinext from "vinext";
import { defineConfig } from "vite";
import hostingConfig from "./.openai/hosting.json";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  main: "./worker/index.ts",
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: "site-creator-r2",
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // `workerd` does not ship a Windows ARM64 binary. Keep local development
  // available on that platform; x64 and hosted builds still use Cloudflare.
  const canRunWorkerd = !(process.platform === "win32" && process.arch === "arm64");
  const cloudflare = canRunWorkerd
    ? (await import("@cloudflare/vite-plugin")).cloudflare
    : null;

  return {
    server: {
      // Team testing uses a Cloudflare Quick Tunnel. The browser talks only to
      // this Vite server; API calls are proxied to the local FastAPI process so
      // the backend never needs a second public tunnel.
      allowedHosts: [".trycloudflare.com"],
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8000",
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/api/, ""),
        },
      },
      ...(isCodexSeatbeltSandbox
        ? { watch: { useFsEvents: false, usePolling: true } }
        : {}),
    },
    plugins: [
      vinext(),
      sites(),
      ...(cloudflare
        ? [
            cloudflare({
              viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
              config: localBindingConfig,
            }),
          ]
        : []),
    ],
  };
});
