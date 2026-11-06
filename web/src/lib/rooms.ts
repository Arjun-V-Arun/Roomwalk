// Build-time access to the room files the capture pipeline publishes.
// Runs on the server during `next build`; the exported site has no server.

import "server-only";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { RoomManifest, RoomSummary } from "@/viewer/types";

const ROOMS = path.join(process.cwd(), "public", "rooms");

export function listRooms(): RoomSummary[] {
  const file = path.join(ROOMS, "index.json");
  if (!existsSync(file)) return [];
  return (JSON.parse(readFileSync(file, "utf8")) as { rooms: RoomSummary[] }).rooms;
}

export function getRoom(id: string): RoomManifest {
  return JSON.parse(readFileSync(path.join(ROOMS, id, "manifest.json"), "utf8")) as RoomManifest;
}
