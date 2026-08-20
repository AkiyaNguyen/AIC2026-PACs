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
  assert.match(page, /query_vi/);
  assert.match(page, /query_en/);
  assert.match(page, /query_en_source/);
  assert.match(page, /Tiếng Anh cho CLIP/);
  assert.match(page, /weight_visual/);
  assert.match(page, /weight_transcript/);
  assert.match(page, /weight_sem_text/);
  assert.match(page, /weight_bm25/);
  assert.match(page, /bm25_top_segments/);
  assert.match(page, /sem_top_segments/);
  assert.match(page, /Nguồn candidate/);
  assert.match(page, /resolveApiUrl/);
  assert.match(page, /thumbnail_url/);
  assert.match(page, /video_url/);
  assert.match(page, /preload="metadata"/);
  assert.match(page, /jumpToKeyframe/);
  assert.match(page, /Về keyframe/);
  assert.match(page, /playbackPosition/);
  assert.match(page, /onTimeUpdate/);
  assert.match(page, /Frame ID hiện tại/);
  assert.match(page, /Chép đáp án tại playhead/);
  assert.match(page, /onError=\{\(\) => markThumbnailFailed/);
  assert.match(page, /Xuất CSV/);
  assert.match(layout, /PACs Search · AIC 2026/);
  assert.doesNotMatch(page, /SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
