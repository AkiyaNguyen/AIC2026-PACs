import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("ships the PACs KIS workspace instead of the starter preview", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("package.json", root), "utf8"),
  ]);

  assert.match(page, /PACs Search/);
  assert.match(page, /\/check_health/);
  assert.match(page, /\/search/);
  assert.match(page, /weight_clip/);
  assert.match(page, /weight_asr/);
  assert.match(page, /Xuất CSV/);
  assert.match(layout, /PACs Search · AIC 2026/);
  assert.doesNotMatch(page, /SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
