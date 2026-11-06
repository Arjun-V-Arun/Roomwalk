// Render a room's start view to a JPEG, for link previews and the home page.
// Real scans already get a poster from the sharpest video frame; this is for
// rooms without one, such as the synthetic demo.
//
//   npm run build && node scripts/poster.mjs demo --out poster.jpg

import { launch, openRoom, startServer, arg } from "./browser.mjs";

const id = process.argv[2];
if (!id || id.startsWith("--")) {
  console.error("usage: node scripts/poster.mjs <room-id> --out poster.jpg [--gpu]");
  process.exit(1);
}
const out = arg("out", `${id}-poster.jpg`);
const { server, base } = await startServer();
const browser = await launch();
try {
  const page = await openRoom(browser, base, id, { width: 1200, height: 900, query: "&tier=high&format=sog" });
  await page.waitForTimeout(4000); // let level of detail settle at full budget
  await page.screenshot({ path: out, type: "jpeg", quality: 85 });
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
  server.close();
}
