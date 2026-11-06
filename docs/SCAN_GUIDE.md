# Scan Guide: From Phone Video to Published Room

This is the full path for one real room, written for a Windows laptop with an NVIDIA GPU. Plan on about an hour the first time: 20 minutes of setup, 2 minutes of filming, and the rest is the computer working.

## Checklist

| Step | Done when |
| --- | --- |
| 1. Install the tools | `roomwalk --help` prints, `npm run build` passes in `web/`, `brush_app.exe --help` prints |
| 2. Issue a code | You have `<room>-code.png` open on a second screen |
| 3. Film the room | A 60 to 90 second video, starting with the code, copied to the laptop as the original file |
| 4. Check the video | `roomwalk check` says PASS or WARN, not FAIL |
| 5. Build the room | `roomwalk build` prints "Published" |
| 6. Look at it on your phone | The room opens from your laptop's address on Wi-Fi |
| 7. Add spots | At least three spots saved into the manifest |
| 8. Record phone numbers | First view time and frame rate from `?stats=1` pasted into `docs/phone-results.md` |

## 1. Install the Tools

Once per machine.

**Python 3.12 and Node.js 20 or newer.** Python 3.12 is the safest choice because `pycolmap` publishes Windows wheels for it.

```powershell
cd roomwalk\capture
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[sfm,plot,dev]"
roomwalk --help

cd ..\web
npm install
npm run build
```

**Brush (training).** Download the Windows zip from the [Brush releases page](https://github.com/ArthurBrussee/brush/releases) (v0.3.0 at the time of writing), unzip it somewhere like `C:\tools\brush`, then tell RoomWalk where it is:

```powershell
setx ROOMWALK_BRUSH "C:\tools\brush\brush_app.exe"
```

Open a new terminal afterwards so the variable is picked up. Brush runs on the GPU through Vulkan or DirectX 12, so no CUDA toolkit is needed.

**build-lod (optional, recommended).** This makes the streamed version of the room, which appears several times sooner on a slow connection (see the benchmark in the README). It needs Rust and the Visual Studio C++ build tools.

```powershell
winget install Rustlang.Rustup
git clone --depth 1 https://github.com/sparkjsdev/spark.git C:\tools\spark
cd C:\tools\spark\rust\build-lod
cargo build --release
setx ROOMWALK_BUILD_LOD "C:\tools\spark\rust\target\release\build-lod.exe"
```

Without it, rooms are published as a single SOG file only, which still works.

**COLMAP binary (optional).** `pycolmap` runs structure from motion on the CPU. The CUDA build of COLMAP from its [releases page](https://github.com/colmap/colmap/releases) is faster; pass its path with `--colmap`.

## 2. Issue a Code

```powershell
roomwalk code my-room
```

This writes `my-room-code.png` and prints the code. It is valid for 48 hours. For your own room, open the PNG on your laptop screen. For someone else's room, send them the PNG.

The signing key lives in `%USERPROFILE%\.roomwalk\secret.key`. Keep it: every later check uses it.

## 3. Film the Room

Before you press record:

- Turn on every light and open the curtains.
- Switch off the ceiling fan. Anything moving turns into a smear.
- Clear people, pets and anything that will move out of the room.
- Use the main camera at 1080p or 4K, 30 fps, in landscape. Not the ultra-wide lens.
- Wipe the lens.
- Tap and hold on a wall to lock focus and exposure.

Then, in one continuous video:

1. **Code first.** Hold the phone close to the screen showing the code for about 2 seconds, so the code fills a good part of the frame.
2. **Loop along the walls.** Walk slowly around the edge of the room, pointing the camera across the room and slightly down. Take about 30 seconds for the loop.
3. **Loop from the middle.** Stand near the centre and turn slowly on the spot, with the phone tilted down enough that the floor fills the bottom third of the frame. Tilt up once for the ceiling. The floor is how the pipeline works out the room's real size, so don't skip it.
4. **Slow down at anything a renter will care about:** the bathroom door, the window, damp patches, the switchboard.

Aim for 60 to 90 seconds in total, about a quarter turn every 5 seconds. Avoid pointing straight at a blank wall; keep a corner, door frame or piece of furniture in each shot.

**Copying the file:** use a USB cable, Google Drive or WhatsApp's "send as document". Normal WhatsApp video compresses the file and removes the recording time, which weakens the freshness check.

## 4. Check the Video

```powershell
roomwalk check C:\scans\my-room.mp4 --room my-room --plot check.png
```

Takes under a minute. You get a verdict and, for each problem, what to do differently. `check.png` shows sharpness, brightness and detail over time, so you can see exactly which seconds were bad.

- **FAIL**: film again. Building from this video will waste 30 minutes.
- **WARN**: fine to build. The warnings are shown to renters as "known gaps".
- **PASS**: build.

## 5. Build the Room

```powershell
roomwalk build C:\scans\my-room.mp4 --room my-room --title "Single room, 2nd floor" --area "Koramangala 5th Block" --rent 9500 --deposit 19000
```

What happens, and roughly how long each stage takes on an RTX 4070 laptop:

| Stage | What it does | Time |
| --- | --- | --- |
| check | Same as step 4 | under 1 min |
| frames | Keeps the sharpest frame in each of 150 time windows | 1 min |
| sfm | Finds where the phone was for each frame | 5 to 15 min on CPU |
| align | Works out up, scale and starting view | seconds |
| train | Trains the Gaussian splat with Brush | 15 to 30 min |
| convert | Cleans floaters, writes SOG and the streamed RAD | 1 to 2 min |
| publish | Copies files into `web/public/rooms/my-room/` | seconds |

Everything is saved in `work\my-room\`. If a stage fails, fix the cause and resume from it instead of starting over:

```powershell
roomwalk build C:\scans\my-room.mp4 --room my-room --from train
```

**If Brush's command line fails** (release versions sometimes rename options), train in the Brush window instead: open `brush_app.exe`, load the folder `work\my-room\dataset`, let it train, export the `.ply`, then continue:

```powershell
roomwalk build --room my-room --ply C:\path\to\export.ply
```

## 6. Look at It on Your Phone

```powershell
cd web
npm run build
npm start
```

Find your laptop's address with `ipconfig` (the IPv4 address, something like `192.168.1.23`). On a phone on the same Wi-Fi, open `http://192.168.1.23:3000/r/my-room/`.

If the room starts tilted or sideways, the cameras disagreed about which way was up. Check the `align` line in the build output: an "up spread" above 15 degrees means the phone was held at odd angles. Filming again in landscape usually fixes it.

If the room feels too big or too small, or you sink into the floor, the floor was never clearly in view. The `align` line says where the scale came from: "camera height above floor" is reliable, "ceiling" is a reasonable estimate, and "assumed typical room width" is a guess. Film again with the floor in view during the middle loop.

## 7. Add Spots

Open `/r/my-room/?edit=1`. Walk to a good view of the bed, press "Save this view" and name it. Repeat for the window, the bathroom door and anything else. Press "Copy spots" and paste the result into `"hotspots"` in `web/public/rooms/my-room/manifest.json`, then rebuild.

To change where the room opens, use "Copy start view" and paste it into `"view": { "start": ... }`.

Rebuilding the room from a new video clears spots, because everything moves. Pass `--keep-spots` only if you re-ran publishing without re-filming.

## 8. Record Phone Numbers

Open `/r/my-room/?stats=1` on your phone. Wait for the room, look around for 10 seconds, then press "Copy numbers" and paste into `docs/phone-results.md`. Do it on mobile data as well as Wi-Fi, and on the cheapest phone you can borrow. These are the numbers that matter in an interview.

## Putting It Online

The site is static files, so any static host works.

- **Vercel:** import the repo, set the root directory to `web`, and set `NEXT_PUBLIC_SITE_URL` to your Vercel address so link previews have absolute image URLs.
- **GitHub Pages:** build with `NEXT_PUBLIC_BASE_PATH=/roomwalk` and publish `web/out`.

A room is roughly 3 to 30 MB depending on splat count, so a handful of rooms fits comfortably in a normal Git repository.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `check` says no code found | Code too small or too brief in frame | Hold it closer for 2 full seconds |
| Freshness says `expired` | Filmed over 48 h after issuing | Issue a new code |
| Freshness weak timestamp note | File was forwarded, not the original | Copy the original file |
| Under 60% of frames placed in sfm | Blur, darkness or blank walls | Film slower with more light and more furniture in view |
| Room is full of holes behind furniture | Furniture only seen from one side | Add the middle loop and slow down there |
| Brush runs out of memory | Too many splats for 8 GB | `--max-splats 1000000` |
| `splat-transform` floater filter fails | No GPU adapter found | Nothing to do: it falls back and skips only the floater filter |
